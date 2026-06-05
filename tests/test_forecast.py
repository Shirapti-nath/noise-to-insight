"""Tests for Phase 4 predictive analytics."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from src.models.forecast import ForecastNarrative, PrescriptiveAction
from src.phases.forecast import (
    NO_TIME_COLUMN_MESSAGE,
    detect_target_column,
    detect_time_column,
    load_forecast,
    run_forecast,
    run_forecast_pipeline,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def forecast_df() -> pl.DataFrame:
    return pl.read_csv(FIXTURES / "forecast_series.csv", try_parse_dates=False).with_columns(
        pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col("amount").cast(pl.Float64),
    )


@pytest.fixture
def no_time_df() -> pl.DataFrame:
    """Numeric-only frame: no date column and no group dimension for snapshot."""
    return pl.DataFrame(
        {
            "amount": [10.0, 20.0, 30.0],
            "units": [1.0, 2.0, 3.0],
        }
    )


def test_detect_time_and_target(forecast_df: pl.DataFrame) -> None:
    time_col = detect_time_column(forecast_df)
    target_col = detect_target_column(forecast_df, time_col)
    assert time_col == "order_date"
    assert target_col == "amount"


def test_run_forecast_pipeline_success(forecast_df: pl.DataFrame) -> None:
    result = run_forecast_pipeline(forecast_df, use_llm=False)

    assert result.status == "success"
    assert result.time_column == "order_date"
    assert result.target_column == "amount"
    assert result.model in ("prophet", "sklearn_linear")
    assert len(result.forecast) == 30
    assert result.forecast[0].lower is not None
    assert result.forecast_narrative
    assert len(result.prescriptive_actions) == 2
    assert result.prescriptive_actions[0].due_date


def test_missing_time_column_graceful(no_time_df: pl.DataFrame) -> None:
    result = run_forecast_pipeline(no_time_df, use_llm=False)

    assert result.status == "skipped"
    assert result.user_message == NO_TIME_COLUMN_MESSAGE
    assert result.forecast == []


def test_snapshot_forecast_for_hr_style_data() -> None:
    df = pl.DataFrame(
        {
            "department": ["Sales", "Sales", "R&D", "R&D"],
            "monthlyincome": [5000.0, 6000.0, 8000.0, 9000.0],
        }
    )
    result = run_forecast_pipeline(df, use_llm=False)
    assert result.status == "snapshot"
    assert result.target_column == "monthlyincome"
    assert len(result.forecast) >= 2
    assert result.forecast_narrative


def test_run_forecast_writes_artifacts(tmp_path: Path, forecast_df: pl.DataFrame) -> None:
    cleaned_path = tmp_path / "cleaned.parquet"
    artifact_dir = tmp_path / "run"
    forecast_df.write_parquet(cleaned_path)

    json_path = run_forecast(cleaned_path, artifact_dir, use_llm=False)
    assert json_path.exists()
    assert (artifact_dir / "forecast.png").exists()

    payload = json.loads(json_path.read_text())
    assert payload["status"] == "success"
    assert payload["chart_path"] == "forecast.png"
    assert len(payload["forecast"]) == 30

    loaded = load_forecast(json_path)
    assert loaded.status == "success"


def test_run_forecast_skipped_writes_json_without_chart(
    tmp_path: Path,
    no_time_df: pl.DataFrame,
) -> None:
    cleaned_path = tmp_path / "cleaned.parquet"
    artifact_dir = tmp_path / "skipped"
    no_time_df.write_parquet(cleaned_path)

    json_path = run_forecast(cleaned_path, artifact_dir, use_llm=False)
    assert json_path.exists()
    assert not (artifact_dir / "forecast.png").exists()

    payload = json.loads(json_path.read_text())
    assert payload["status"] == "skipped"
    assert NO_TIME_COLUMN_MESSAGE in payload["user_message"]


@patch("src.phases.forecast.generate_forecast_narrative_llm")
def test_run_forecast_with_mock_llm(
    mock_narrative,
    tmp_path: Path,
    forecast_df: pl.DataFrame,
) -> None:
    mock_narrative.return_value = (
        "Revenue is trending up 8% over the next month.",
        [
            PrescriptiveAction(action="Increase safety stock.", due_date="2026-06-15"),
            PrescriptiveAction(action="Review supplier SLAs.", due_date="2026-06-22"),
        ],
        "llm",
    )

    cleaned_path = tmp_path / "cleaned.parquet"
    artifact_dir = tmp_path / "run_llm"
    forecast_df.write_parquet(cleaned_path)

    run_forecast(cleaned_path, artifact_dir, use_llm=True)
    mock_narrative.assert_called_once()

    payload = json.loads((artifact_dir / "forecast.json").read_text())
    assert payload["narrative_source"] == "llm"
    assert "trending" in payload["forecast_narrative"]
