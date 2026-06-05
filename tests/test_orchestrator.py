"""Tests for LangGraph orchestrator and end-to-end pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ARTIFACTS_DIR
from src.orchestrator.graph import PHASE_ORDER, build_pipeline_graph, run_pipeline
from src.phases.report import load_report_meta

FIXTURES = Path(__file__).parent / "fixtures"


def test_build_pipeline_graph_compiles() -> None:
    graph = build_pipeline_graph()
    assert graph is not None


def test_run_pipeline_end_to_end(tmp_path: Path) -> None:
    """Full pipeline without LLM on segment orders."""
    segment = FIXTURES / "segment_orders.csv"
    run_id = "test_e2e"
    final = run_pipeline([segment], run_id, use_llm=False)

    artifact_dir = final.artifact_dir
    assert artifact_dir.exists()
    assert (artifact_dir / "profile.json").exists()
    assert (artifact_dir / "cleaned.parquet").exists()
    assert (artifact_dir / "patterns.json").exists()
    assert (artifact_dir / "anomalies.json").exists()
    assert (artifact_dir / "forecast.json").exists()
    assert (artifact_dir / "graph.json").exists()
    assert (artifact_dir / "graph.html").exists()
    assert (artifact_dir / "report.html").exists()

    assert len(final.phase_log) == len(PHASE_ORDER)
    assert final.headline_insight
    assert final.error is None

    meta = load_report_meta(artifact_dir)
    assert meta.get("executive_summary")


def test_run_pipeline_with_forecast_series(tmp_path: Path) -> None:
    """Forecast phase succeeds with 12-day series."""
    path = FIXTURES / "forecast_series.csv"
    final = run_pipeline([path], "test_forecast_e2e", use_llm=False)
    forecast = final.artifact_dir / "forecast.json"
    assert forecast.exists()
    import json

    data = json.loads(forecast.read_text())
    assert data["status"] == "success"
