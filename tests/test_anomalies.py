"""Tests for Phase 3 anomaly detection."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.models.state import AnomalyRecord
from src.models.anomalies import ExplainedAnomalies
from src.phases.anomalies import (
    align_anomaly_record,
    build_graph_entity_id,
    detect_numeric_anomalies,
    discover_anomalies,
    get_anomalies_for_entity,
    link_anomaly_to_graph_entity,
    load_anomalies,
    run_anomalies,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def cleaned_anomaly_df() -> pl.DataFrame:
    return pl.read_csv(FIXTURES / "anomaly_orders.csv", try_parse_dates=False).with_columns(
        pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col("amount").cast(pl.Float64),
    )


@pytest.fixture
def support_df() -> pl.DataFrame:
    return pl.read_csv(FIXTURES / "support_tickets.csv")


def test_build_graph_entity_id() -> None:
    assert build_graph_entity_id("order_id", "ORD-9") == "order:ORD-9"
    assert build_graph_entity_id("supplier_id", "SUP-9") == "supplier:SUP-9"


def test_link_anomaly_to_graph_entity() -> None:
    record = AnomalyRecord(entity="ORD-9", metric="amount", score=0.9)
    linked = link_anomaly_to_graph_entity(
        record,
        entity_column="order_id",
        entity_value="ORD-9",
    )
    assert linked.graph_entity_id == "order:ORD-9"


def test_detect_numeric_anomalies_finds_outlier(cleaned_anomaly_df: pl.DataFrame) -> None:
    candidates = detect_numeric_anomalies(cleaned_anomaly_df, top_n=3)
    assert candidates
    top = candidates[0]
    assert top["entity"] == "ORD-9"
    assert top["metric"] == "amount"
    assert top["score"] >= 0.5


def test_align_anomaly_record_restores_entity_from_candidate(
    cleaned_anomaly_df: pl.DataFrame,
) -> None:
    candidates = detect_numeric_anomalies(cleaned_anomaly_df, top_n=1)
    llm_record = AnomalyRecord(
        entity="unknown",
        metric=candidates[0]["metric"],
        score=candidates[0]["score"],
        severity="critical",
        hypothesis="Multivariate outlier for unknown.",
        recommended_action="Investigate unknown.",
    )
    aligned = align_anomaly_record(llm_record, candidates, cleaned_anomaly_df)
    assert aligned.entity == "ORD-9"
    assert "unknown" not in (aligned.hypothesis or "").lower()


def test_discover_anomalies_heuristic(cleaned_anomaly_df: pl.DataFrame) -> None:
    result = discover_anomalies(cleaned_anomaly_df, use_llm=False, top_n=5)

    assert result.anomalies
    assert result.anomalies[0].score >= result.anomalies[-1].score
    assert result.anomalies[0].hypothesis
    assert result.anomalies[0].recommended_action
    assert result.anomalies[0].graph_entity_id == "order:ORD-9"
    assert "order:ORD-9" in result.entity_index


def test_text_heuristic_on_support(support_df: pl.DataFrame) -> None:
    result = discover_anomalies(support_df, use_llm=False, top_n=5)
    text_records = [a for a in result.anomalies if a.source.startswith("text")]
    assert text_records
    assert any("reason" in a.metric for a in text_records)


def test_run_anomalies_writes_json(tmp_path: Path, cleaned_anomaly_df: pl.DataFrame) -> None:
    cleaned_path = tmp_path / "cleaned.parquet"
    artifact_dir = tmp_path / "run"
    cleaned_anomaly_df.write_parquet(cleaned_path)

    out = run_anomalies(cleaned_path, artifact_dir, use_llm=False)
    assert out == artifact_dir / "anomalies.json"

    payload = json.loads(out.read_text())
    assert payload["anomaly_count"] >= 1

    loaded = load_anomalies(out)
    hits = get_anomalies_for_entity(loaded, "order:ORD-9")
    assert hits
    assert hits[0].entity == "ORD-9"


@patch("src.phases.anomalies.explain_anomalies_llm")
def test_run_anomalies_with_mock_llm(
    mock_explain: MagicMock,
    tmp_path: Path,
    cleaned_anomaly_df: pl.DataFrame,
) -> None:
    mock_explain.return_value = (
        [
            AnomalyRecord(
                entity="ORD-9",
                metric="amount",
                score=0.95,
                severity="critical",
                hypothesis="Single order amount spike vs regional baseline.",
                recommended_action="Audit ORD-9 pricing and fulfillment.",
                graph_entity_id="order:ORD-9",
                source="isolation_forest",
            )
        ],
        "llm",
    )

    cleaned_path = tmp_path / "cleaned.parquet"
    artifact_dir = tmp_path / "run_llm"
    cleaned_anomaly_df.write_parquet(cleaned_path)

    run_anomalies(cleaned_path, artifact_dir, use_llm=True)
    mock_explain.assert_called_once()
    report = json.loads((artifact_dir / "anomalies.json").read_text())
    assert report["explanation_source"] == "llm"
    assert report["anomalies"][0]["severity"] == "critical"
