"""Tests for Phase 2 pattern discovery."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.models.state import InsightCard
from src.phases.patterns import (
    compute_segment_lifts,
    discover_raw_patterns,
    get_top_insights,
    load_patterns,
    run_pattern_discovery,
    run_patterns,
)

FIXTURES = Path(__file__).parent / "fixtures"
SEGMENT_CSV = FIXTURES / "segment_orders.csv"


@pytest.fixture
def cleaned_segment_df() -> pl.DataFrame:
    return pl.read_csv(SEGMENT_CSV, try_parse_dates=False).with_columns(
        pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col("amount").cast(pl.Float64),
    )


def test_compute_correlations_polars_matrix(cleaned_segment_df: pl.DataFrame) -> None:
    """Regression: Polars 1.x corr() has no 'column' field."""
    from src.phases.patterns import compute_correlations

    df = cleaned_segment_df.select(["amount", "order_date"]).with_columns(
        pl.lit(1).alias("units"),
        pl.lit(2).alias("extra_num"),
    )
    correlations = compute_correlations(df)
    assert isinstance(correlations, list)


def test_discover_raw_patterns_includes_segment_lift(cleaned_segment_df: pl.DataFrame) -> None:
    statistics, raw = discover_raw_patterns(cleaned_segment_df)

    assert statistics["row_count"] == 8
    assert statistics["pattern_counts"]["segment_lift"] >= 1
    types = {p["type"] for p in raw}
    assert "segment_lift" in types or "time_trend" in types


def test_compute_segment_lifts_east_higher_amount(cleaned_segment_df: pl.DataFrame) -> None:
    lifts = compute_segment_lifts(cleaned_segment_df)
    region_lift = next((p for p in lifts if p.get("segment_column") == "region"), None)
    assert region_lift is not None
    assert region_lift["segment_value"] == "East"
    assert region_lift["segment_mean"] > region_lift["global_mean"]


def test_run_pattern_discovery_heuristic(cleaned_segment_df: pl.DataFrame) -> None:
    result = run_pattern_discovery(cleaned_segment_df, use_llm=False)

    assert 1 <= len(result.insights) <= 5
    assert result.insight_source == "heuristic"
    assert result.raw_patterns
    assert all(isinstance(i, InsightCard) for i in result.insights)


def test_run_patterns_writes_json(tmp_path: Path, cleaned_segment_df: pl.DataFrame) -> None:
    cleaned_path = tmp_path / "cleaned.parquet"
    artifact_dir = tmp_path / "run"
    cleaned_segment_df.write_parquet(cleaned_path)

    out = run_patterns(cleaned_path, artifact_dir, use_llm=False)
    assert out == artifact_dir / "patterns.json"

    payload = json.loads(out.read_text())
    assert payload["row_count"] == 8
    assert "insights" in payload
    assert "statistics" in payload
    assert "generated_at" in payload


def test_get_top_insights(tmp_path: Path, cleaned_segment_df: pl.DataFrame) -> None:
    result = run_pattern_discovery(cleaned_segment_df, use_llm=False)
    top2 = get_top_insights(result, n=2)
    assert len(top2) == 2

    patterns_path = tmp_path / "patterns.json"
    patterns_path.write_text(
        json.dumps({**result.model_dump(mode="json"), "generated_at": "2024-01-01"}),
    )
    loaded_top = get_top_insights(patterns_path, n=3)
    assert len(loaded_top) == 3


@patch("src.phases.patterns.rank_insights_llm")
def test_run_patterns_with_mock_llm(
    mock_rank: MagicMock,
    tmp_path: Path,
    cleaned_segment_df: pl.DataFrame,
) -> None:
    mock_insights = [
        InsightCard(
            title="East region premium",
            summary="East orders average 40%+ higher amount than West.",
            impact="high",
            evidence={"region": "East", "lift_pct": 42.0},
        )
    ]
    mock_rank.return_value = (mock_insights, "llm")

    cleaned_path = tmp_path / "cleaned.parquet"
    artifact_dir = tmp_path / "run_llm"
    cleaned_segment_df.write_parquet(cleaned_path)

    run_patterns(cleaned_path, artifact_dir, use_llm=True)
    mock_rank.assert_called_once()

    report = json.loads((artifact_dir / "patterns.json").read_text())
    assert report["insight_source"] == "llm"
    assert report["insights"][0]["title"] == "East region premium"


def test_integration_cleaning_then_patterns(tmp_path: Path) -> None:
    """End-to-end from messy CSV through cleaning into patterns."""
    from src.ingest.profiler import profile_dataset
    from src.models.cleaning import (
        CleaningPlan,
        ColumnRename,
        DateParser,
        DtypeFix,
        FillStrategy,
    )
    from src.phases.cleaning import run_cleaning

    messy = FIXTURES / "messy_orders.csv"
    profile_path = tmp_path / "profile.json"
    profile_dataset([messy], profile_path)

    plan = CleaningPlan(
        column_renames=[ColumnRename(source="order id", target="order_id")],
        currency_columns=["amount"],
        date_parsers=[DateParser(column="order_date")],
        dtype_fixes=[DtypeFix(column="amount", target_dtype="float")],
        fill_strategies=[FillStrategy(column="amount", strategy="median")],
        dedup_keys=["order_id"],
        required_columns=["order_id", "amount", "order_date"],
    )
    artifact = tmp_path / "phase1"
    cleaned = run_cleaning(profile_path, artifact, plan=plan, use_llm=False, plan_source="provided")
    patterns_path = run_patterns(cleaned, tmp_path / "phase2", use_llm=False)

    data = load_patterns(patterns_path)
    assert data.insights
    assert data.row_count == 3
