"""Tests for Phase 1 data cleaning."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.ingest.profiler import profile_dataset
from src.models.cleaning import (
    CleaningPlan,
    ColumnRename,
    DateParser,
    DtypeFix,
    FillStrategy,
)
from src.phases.cleaning import (
    CleaningValidationError,
    build_heuristic_cleaning_plan,
    execute_cleaning_plan,
    run_cleaning,
    validate_cleaned,
)

FIXTURES = Path(__file__).parent / "fixtures"
MESSY_CSV = FIXTURES / "messy_orders.csv"


def _messy_cleaning_plan() -> CleaningPlan:
    return CleaningPlan(
        column_renames=[ColumnRename(source="order id", target="order_id")],
        currency_columns=["amount"],
        date_parsers=[DateParser(column="order_date")],
        dtype_fixes=[DtypeFix(column="amount", target_dtype="float")],
        fill_strategies=[FillStrategy(column="amount", strategy="median")],
        dedup_keys=["order_id"],
        required_columns=["order_id", "amount", "order_date"],
        max_duplicate_rate=0.05,
    )


def test_execute_messy_csv_removes_currency_and_dedup() -> None:
    from src.ingest.loader import load_file

    raw = load_file(MESSY_CSV)
    plan = _messy_cleaning_plan()
    cleaned, rows_before = execute_cleaning_plan({"messy_orders.csv": raw}, plan)

    assert rows_before == 4
    assert cleaned.height == 3
    assert "order_id" in cleaned.columns
    assert cleaned["amount"].null_count() == 0
    assert cleaned["amount"].dtype in (pl.Float32, pl.Float64)
    assert cleaned.filter(pl.col("order_id") == "ORD-1").height == 1
    assert cleaned["amount"].min() >= 15.0


def test_validate_cleaned_passes_for_messy_plan() -> None:
    from src.ingest.loader import load_file

    raw = load_file(MESSY_CSV)
    plan = _messy_cleaning_plan()
    cleaned, rows_before = execute_cleaning_plan({"messy_orders.csv": raw}, plan)
    result = validate_cleaned(cleaned, plan, rows_before)

    assert result.passed is True
    assert result.duplicate_rate <= plan.max_duplicate_rate
    assert result.missing_columns == []


def test_validate_fails_when_duplicate_rate_exceeded() -> None:
    df = pl.DataFrame(
        {
            "order_id": ["ORD-1", "ORD-1", "ORD-2"],
            "amount": [10.0, 20.0, 30.0],
            "order_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )
    plan = CleaningPlan(
        dedup_keys=["order_id"],
        required_columns=["order_id", "amount"],
        max_duplicate_rate=0.01,
    )
    result = validate_cleaned(df, plan, rows_before=3)

    assert result.passed is False
    assert result.duplicate_rate > plan.max_duplicate_rate


def test_heuristic_plan_from_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_dataset([MESSY_CSV], profile_path)
    profile = json.loads(profile_path.read_text())
    from src.ingest.loader import load_files

    frames = load_files([MESSY_CSV])
    plan = build_heuristic_cleaning_plan(profile, frames)

    assert any(r.target == "order_id" for r in plan.column_renames)
    assert "amount" in plan.currency_columns
    assert "order_id" in plan.dedup_keys or "order_id" in plan.required_columns


def test_run_cleaning_writes_artifacts(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_dataset([MESSY_CSV], profile_path)
    artifact_dir = tmp_path / "run1"

    parquet_path = run_cleaning(
        profile_path,
        artifact_dir,
        plan=_messy_cleaning_plan(),
        use_llm=False,
        plan_source="provided",
    )

    assert parquet_path.exists()
    report_path = artifact_dir / "cleaning_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text())
    assert report["status"] == "passed"
    assert report["plan_source"] == "provided"
    assert report["validation"]["rows_after"] == 3

    df = pl.read_parquet(parquet_path)
    assert df.height == 3


def test_run_cleaning_raises_on_validation_failure(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_dataset([MESSY_CSV], profile_path)
    bad_plan = _messy_cleaning_plan().model_copy(
        update={"dedup_keys": [], "required_columns": ["nonexistent_column"]},
    )

    with pytest.raises(CleaningValidationError):
        run_cleaning(
            profile_path,
            tmp_path / "run_fail",
            plan=bad_plan,
            use_llm=False,
            plan_source="provided",
        )

    report = json.loads((tmp_path / "run_fail" / "cleaning_report.json").read_text())
    assert report["status"] == "failed"


@patch("src.phases.cleaning.create_cleaning_plan_llm")
def test_run_cleaning_with_mock_llm(
    mock_llm: MagicMock,
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.json"
    profile_dataset([MESSY_CSV], profile_path)
    mock_llm.return_value = (_messy_cleaning_plan(), "llm")

    artifact_dir = tmp_path / "run_llm"
    run_cleaning(profile_path, artifact_dir, use_llm=True)

    mock_llm.assert_called_once()
    report = json.loads((artifact_dir / "cleaning_report.json").read_text())
    assert report["status"] == "passed"
    assert report["plan_source"] == "llm"
