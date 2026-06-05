"""Pydantic models for Phase 1 data cleaning."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ColumnRename(BaseModel):
    """Map a raw column name to a canonical name."""

    source: str = Field(description="Original column name as it appears in the file")
    target: str = Field(description="Canonical column name after cleaning")


class DtypeFix(BaseModel):
    """Target dtype for a column after cleaning transforms."""

    column: str
    target_dtype: Literal["string", "int", "float", "boolean", "date"]


class DateParser(BaseModel):
    """Date parsing rules for a column with inconsistent formats."""

    column: str
    formats: list[str] = Field(
        default_factory=lambda: [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%m-%d-%Y",
        ],
        description="strftime formats to try in order (coalesced)",
    )


class FillStrategy(BaseModel):
    """Missing-value handling for a column."""

    column: str
    strategy: Literal["mean", "median", "zero", "mode", "forward_fill", "drop", "constant"] = "median"
    constant_value: str | float | int | None = None


class CleaningPlan(BaseModel):
    """Structured cleaning plan produced by the LLM or heuristics."""

    column_renames: list[ColumnRename] = Field(default_factory=list)
    dtype_fixes: list[DtypeFix] = Field(default_factory=list)
    dedup_keys: list[str] = Field(
        default_factory=list,
        description="Columns that define a logical record for duplicate detection",
    )
    date_parsers: list[DateParser] = Field(default_factory=list)
    fill_strategies: list[FillStrategy] = Field(default_factory=list)
    currency_columns: list[str] = Field(
        default_factory=list,
        description="Columns containing currency symbols to strip before numeric cast",
    )
    required_columns: list[str] = Field(
        default_factory=list,
        description="Columns that must exist after cleaning",
    )
    max_duplicate_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum allowed duplicate rate on dedup_keys (0.05 = 5%)",
    )
    join_keys: list[str] = Field(
        default_factory=list,
        description="Columns used to join multiple input files",
    )
    primary_file: str | None = Field(
        default=None,
        description="Primary table filename when joining multiple files",
    )


class CleaningValidation(BaseModel):
    """Post-clean validation results."""

    passed: bool
    missing_columns: list[str] = Field(default_factory=list)
    duplicate_rate: float = 0.0
    rows_before: int = 0
    rows_after: int = 0
    message: str = ""


class CleaningReport(BaseModel):
    """Written to cleaning_report.json."""

    status: Literal["passed", "failed"]
    plan_source: Literal["llm", "heuristic", "provided"]
    input_files: list[str]
    validation: CleaningValidation
    plan: CleaningPlan
