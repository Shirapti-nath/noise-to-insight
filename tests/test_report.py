"""Tests for Phase 6 executive report."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from src.models.cleaning import (
    CleaningPlan,
    ColumnRename,
    DateParser,
    DtypeFix,
    FillStrategy,
)
from src.models.report import ExecutiveReportContent
from src.phases.cleaning import run_cleaning
from src.phases.forecast import run_forecast
from src.phases.graph_build import run_graph_build
from src.phases.anomalies import run_anomalies
from src.phases.patterns import run_patterns
from src.phases.report import (
    build_heuristic_executive_content,
    build_template_context,
    gather_artifact_context,
    render_report_html,
    run_report,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def pipeline_artifacts(tmp_path: Path) -> Path:
    """Run phases 1–5 on segment data into a single artifact dir."""
    from src.ingest.profiler import profile_dataset

    segment = FIXTURES / "segment_orders.csv"
    profile_path = tmp_path / "profile.json"
    profile_dataset([segment], profile_path)

    plan = CleaningPlan(
        column_renames=[],
        date_parsers=[DateParser(column="order_date")],
        dtype_fixes=[DtypeFix(column="amount", target_dtype="float")],
        dedup_keys=["order_id"],
        required_columns=["order_id", "amount", "order_date"],
    )
    artifact_dir = tmp_path / "run"
    run_cleaning(profile_path, artifact_dir, plan=plan, use_llm=False, plan_source="provided")
    cleaned = artifact_dir / "cleaned.parquet"

    run_patterns(cleaned, artifact_dir, use_llm=False)
    run_anomalies(cleaned, artifact_dir, use_llm=False)
    run_forecast(cleaned, artifact_dir, use_llm=False)
    run_graph_build(
        cleaned,
        artifact_dir,
        patterns_path=artifact_dir / "patterns.json",
        anomalies_path=artifact_dir / "anomalies.json",
        use_llm=False,
    )
    return artifact_dir


def test_gather_artifact_context(pipeline_artifacts: Path) -> None:
    ctx = gather_artifact_context(pipeline_artifacts)
    assert ctx["patterns"] is not None
    assert ctx["anomalies"] is not None
    assert ctx["forecast"] is not None
    assert ctx["graph"] is not None


def test_build_heuristic_executive_content(pipeline_artifacts: Path) -> None:
    ctx = gather_artifact_context(pipeline_artifacts)
    content = build_heuristic_executive_content(ctx)
    assert len(content.executive_summary) > 50
    assert len(content.top_actions) >= 1


def test_render_report_html(pipeline_artifacts: Path, tmp_path: Path) -> None:
    ctx = gather_artifact_context(pipeline_artifacts)
    content = build_heuristic_executive_content(ctx)
    template_ctx = build_template_context(ctx, content)
    html_path = render_report_html(template_ctx, tmp_path / "preview.html")
    html = html_path.read_text()
    assert "Executive Summary" in html
    assert "Top Insights" in html


@patch("src.phases.report.render_report_pdf", return_value=True)
@patch("src.phases.report.generate_executive_content_llm")
def test_run_report_writes_files(
    mock_llm,
    mock_pdf,
    pipeline_artifacts: Path,
) -> None:
    mock_llm.return_value = (
        ExecutiveReportContent(
            executive_summary="Operations data shows East region driving margin risk.",
            top_actions=["Audit East region SKUs", "Expedite supplier review by June 15"],
        ),
        "llm",
    )

    result_path = run_report(pipeline_artifacts, use_llm=True)
    mock_llm.assert_called_once()
    mock_pdf.assert_called_once()

    assert (pipeline_artifacts / "report.html").exists()
    assert (pipeline_artifacts / "report_meta.json").exists()
    meta = json.loads((pipeline_artifacts / "report_meta.json").read_text())
    assert meta["narrative_source"] == "llm"
    assert result_path.name in ("executive_report.pdf", "report.html")


@patch("src.phases.report.render_report_pdf", return_value=False)
def test_run_report_fallback_html_only(mock_pdf, pipeline_artifacts: Path) -> None:
    path = run_report(pipeline_artifacts, use_llm=False)
    assert path.name == "report.html"
    meta = json.loads((pipeline_artifacts / "report_meta.json").read_text())
    assert meta["pdf_generated"] is False
