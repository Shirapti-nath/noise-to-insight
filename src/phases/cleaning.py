"""Phase 1: AI-assisted data cleaning."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import ValidationError

from src.config import get_settings
from src.ingest.loader import load_files
from src.llm.client import get_client, get_deployment_name
from src.models.cleaning import (
    CleaningPlan,
    CleaningReport,
    CleaningValidation,
    ColumnRename,
    DateParser,
    DtypeFix,
    FillStrategy,
)

CURRENCY_PATTERN = re.compile(r"[\$,]")
DATE_NAME_PATTERN = re.compile(r"date|time|timestamp", re.IGNORECASE)
MONEY_NAME_PATTERN = re.compile(r"amount|price|cost|revenue|total|value", re.IGNORECASE)
NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


class CleaningValidationError(Exception):
    """Raised when cleaned data fails validation checks."""


def _normalize_column_name(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())


def _load_profile(profile_path: Path) -> dict[str, Any]:
    return json.loads(profile_path.read_text(encoding="utf-8"))


def _paths_from_profile(profile: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for meta in profile.get("files", {}).values():
        paths.append(Path(meta["path"]))
    if not paths:
        raise ValueError("profile.json contains no file paths")
    return paths


def _rename_map(plan: CleaningPlan) -> dict[str, str]:
    return {r.source: r.target for r in plan.column_renames}


def _apply_renames(df: pl.DataFrame, plan: CleaningPlan) -> pl.DataFrame:
    mapping = _rename_map(plan)
    if not mapping:
        return df
    existing = {src: tgt for src, tgt in mapping.items() if src in df.columns}
    if existing:
        return df.rename(existing)
    return df


def _apply_currency_strip(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for col in columns:
        if col not in df.columns:
            continue
        exprs.append(
            pl.col(col)
            .cast(pl.Utf8, strict=False)
            .str.replace_all(r"[\$,]", "")
            .str.strip_chars()
            .cast(pl.Float64, strict=False)
            .alias(col)
        )
    return df.with_columns(exprs) if exprs else df


def _parse_date_column(df: pl.DataFrame, parser: DateParser) -> pl.DataFrame:
    col = parser.column
    if col not in df.columns:
        return df

    series = df[col]
    if series.dtype in (pl.Date, pl.Datetime):
        return df.with_columns(pl.col(col).cast(pl.Date).alias(col))

    base = pl.col(col).cast(pl.Utf8, strict=False).str.strip_chars()
    attempts = [base.str.strptime(pl.Date, fmt, strict=False) for fmt in parser.formats]
    if not attempts:
        return df
    parsed = pl.coalesce(attempts)
    return df.with_columns(parsed.alias(col))


def _apply_date_parsers(df: pl.DataFrame, plan: CleaningPlan) -> pl.DataFrame:
    for parser in plan.date_parsers:
        df = _parse_date_column(df, parser)
    return df


def _cast_dtype(series: pl.Series, target: str) -> pl.Series:
    if target == "string":
        return series.cast(pl.Utf8, strict=False)
    if target == "int":
        return series.cast(pl.Int64, strict=False)
    if target == "float":
        return series.cast(pl.Float64, strict=False)
    if target == "boolean":
        return series.cast(pl.Boolean, strict=False)
    if target == "date":
        if series.dtype in (pl.Date, pl.Datetime):
            return series.cast(pl.Date, strict=False)
        return series.cast(pl.Utf8, strict=False).str.strptime(pl.Date, strict=False)
    return series


def _apply_dtype_fixes(df: pl.DataFrame, plan: CleaningPlan) -> pl.DataFrame:
    for fix in plan.dtype_fixes:
        if fix.column not in df.columns:
            continue
        df = df.with_columns(_cast_dtype(df[fix.column], fix.target_dtype).alias(fix.column))
    return df


def _apply_fill_strategies(df: pl.DataFrame, plan: CleaningPlan) -> pl.DataFrame:
    for fill in plan.fill_strategies:
        col = fill.column
        if col not in df.columns:
            continue
        series = df[col]
        strategy = fill.strategy
        if strategy == "drop":
            df = df.filter(pl.col(col).is_not_null())
            continue
        if strategy == "zero" and series.dtype in NUMERIC_DTYPES:
            df = df.with_columns(pl.col(col).fill_null(0).alias(col))
        elif strategy == "mean" and series.dtype in NUMERIC_DTYPES:
            df = df.with_columns(pl.col(col).fill_null(pl.col(col).mean()).alias(col))
        elif strategy == "median" and series.dtype in NUMERIC_DTYPES:
            df = df.with_columns(pl.col(col).fill_null(pl.col(col).median()).alias(col))
        elif strategy == "mode":
            mode_val = series.mode().item() if series.null_count() else None
            if mode_val is not None:
                df = df.with_columns(pl.col(col).fill_null(mode_val).alias(col))
        elif strategy == "forward_fill":
            df = df.with_columns(pl.col(col).forward_fill().alias(col))
        elif strategy == "constant":
            df = df.with_columns(pl.col(col).fill_null(fill.constant_value).alias(col))
    return df


def _deduplicate(df: pl.DataFrame, plan: CleaningPlan) -> pl.DataFrame:
    keys = [k for k in plan.dedup_keys if k in df.columns]
    if not keys:
        return df
    return df.unique(subset=keys, keep="last", maintain_order=True)


def _duplicate_rate(df: pl.DataFrame, keys: list[str]) -> float:
    valid_keys = [k for k in keys if k in df.columns]
    if df.height == 0 or not valid_keys:
        return 0.0
    unique_rows = df.select(valid_keys).unique().height
    return round(1.0 - (unique_rows / df.height), 4)


def _join_tables(
    frames: dict[str, pl.DataFrame],
    plan: CleaningPlan,
) -> pl.DataFrame:
    if len(frames) == 1:
        return next(iter(frames.values()))

    names = list(frames.keys())
    primary = plan.primary_file if plan.primary_file in frames else names[0]
    base = frames[primary].with_columns(pl.lit(primary).alias("_source_file"))

    join_keys = [k for k in plan.join_keys if k in base.columns]
    if not join_keys:
        parts = [
            frames[n].with_columns(pl.lit(n).alias("_source_file")) for n in names
        ]
        return pl.concat(parts, how="diagonal_relaxed")

    result = base
    for name, df in frames.items():
        if name == primary:
            continue
        right = df.with_columns(pl.lit(name).alias("_source_file"))
        overlap = [k for k in join_keys if k in right.columns]
        if not overlap:
            continue
        result = result.join(right, on=overlap, how="left", suffix=f"_{name}")
    return result


def execute_cleaning_plan(
    frames: dict[str, pl.DataFrame],
    plan: CleaningPlan,
) -> tuple[pl.DataFrame, int]:
    """Apply plan to raw frames; return cleaned frame and row count before dedup."""
    cleaned_frames: dict[str, pl.DataFrame] = {}
    rows_before = 0

    currency_cols = set(plan.currency_columns)
    for name, df in frames.items():
        rows_before += df.height
        working = _apply_renames(df, plan)
        working = _apply_currency_strip(working, list(currency_cols))
        working = _apply_date_parsers(working, plan)
        working = _apply_dtype_fixes(working, plan)
        working = _apply_fill_strategies(working, plan)
        cleaned_frames[name] = working

    combined = _join_tables(cleaned_frames, plan)
    combined = _deduplicate(combined, plan)
    return combined, rows_before


def validate_cleaned(df: pl.DataFrame, plan: CleaningPlan, rows_before: int) -> CleaningValidation:
    """Validate required columns and duplicate rate."""
    missing = [c for c in plan.required_columns if c not in df.columns]
    dup_rate = _duplicate_rate(df, plan.dedup_keys)
    passed = not missing and dup_rate <= plan.max_duplicate_rate
    message_parts: list[str] = []
    if missing:
        message_parts.append(f"Missing required columns: {missing}")
    if dup_rate > plan.max_duplicate_rate:
        message_parts.append(
            f"Duplicate rate {dup_rate:.2%} exceeds max {plan.max_duplicate_rate:.2%}",
        )
    return CleaningValidation(
        passed=passed,
        missing_columns=missing,
        duplicate_rate=dup_rate,
        rows_before=rows_before,
        rows_after=df.height,
        message="; ".join(message_parts) if message_parts else "OK",
    )


def _sample_has_currency(profile_file: dict[str, Any], column: str) -> bool:
    for row in profile_file.get("sample_rows", []):
        val = row.get(column)
        if isinstance(val, str) and CURRENCY_PATTERN.search(val):
            return True
    return False


def build_heuristic_cleaning_plan(
    profile: dict[str, Any],
    frames: dict[str, pl.DataFrame],
) -> CleaningPlan:
    """Build a reasonable cleaning plan without calling the LLM (offline/tests)."""
    renames: list[ColumnRename] = []
    currency_columns: list[str] = []
    date_parsers: list[DateParser] = []
    dtype_fixes: list[DtypeFix] = []
    fill_strategies: list[FillStrategy] = []
    join_key_sets: list[set[str]] = []

    for file_name, file_profile in profile.get("files", {}).items():
        df = frames.get(file_name)
        if df is None:
            continue
        join_key_sets.append(set(file_profile.get("candidate_join_keys", [])))

        for col in df.columns:
            canonical = _normalize_column_name(col)
            if canonical != col:
                renames.append(ColumnRename(source=col, target=canonical))

            col_key = canonical
            if MONEY_NAME_PATTERN.search(col_key) or _sample_has_currency(file_profile, col):
                if col_key not in currency_columns:
                    currency_columns.append(col_key)
                dtype_fixes.append(DtypeFix(column=col_key, target_dtype="float"))
                fill_strategies.append(FillStrategy(column=col_key, strategy="median"))

            if DATE_NAME_PATTERN.search(col_key):
                date_parsers.append(DateParser(column=col_key))
                dtype_fixes.append(DtypeFix(column=col_key, target_dtype="date"))

    join_keys: list[str] = []
    if join_key_sets:
        common = set.intersection(*join_key_sets) if len(join_key_sets) > 1 else join_key_sets[0]
        join_keys = sorted(common)
        if not join_keys and join_key_sets:
            join_keys = sorted(next(iter(join_key_sets)))

    dedup_keys = join_keys[:] if join_keys else []
    if not dedup_keys:
        first_file = next(iter(profile.get("files", {}).values()), {})
        dedup_keys = first_file.get("candidate_join_keys", [])[:1]

    # Apply renames to dedup/join key names
    rename_lookup = {r.source: r.target for r in renames}
    dedup_keys = [rename_lookup.get(k, _normalize_column_name(k)) for k in dedup_keys]
    join_keys = [rename_lookup.get(k, _normalize_column_name(k)) for k in join_keys]
    currency_columns = [rename_lookup.get(c, _normalize_column_name(c)) for c in currency_columns]

    required = list(dict.fromkeys([*dedup_keys, *join_keys, *currency_columns]))

    primary_file = next(iter(profile.get("files", {}).keys()), None)

    return CleaningPlan(
        column_renames=renames,
        dtype_fixes=_dedupe_dtype_fixes(dtype_fixes),
        dedup_keys=dedup_keys,
        date_parsers=_dedupe_date_parsers(date_parsers),
        fill_strategies=_dedupe_fill_strategies(fill_strategies),
        currency_columns=list(dict.fromkeys(currency_columns)),
        required_columns=required,
        join_keys=join_keys,
        primary_file=primary_file,
    )


def _dedupe_dtype_fixes(fixes: list[DtypeFix]) -> list[DtypeFix]:
    seen: set[str] = set()
    out: list[DtypeFix] = []
    for fix in fixes:
        if fix.column not in seen:
            seen.add(fix.column)
            out.append(fix)
    return out


def _dedupe_date_parsers(parsers: list[DateParser]) -> list[DateParser]:
    seen: set[str] = set()
    out: list[DateParser] = []
    for parser in parsers:
        if parser.column not in seen:
            seen.add(parser.column)
            out.append(parser)
    return out


def _dedupe_fill_strategies(strategies: list[FillStrategy]) -> list[FillStrategy]:
    seen: set[str] = set()
    out: list[FillStrategy] = []
    for strategy in strategies:
        if strategy.column not in seen:
            seen.add(strategy.column)
            out.append(strategy)
    return out


def _profile_prompt_payload(profile: dict[str, Any]) -> str:
    """Compact profile for LLM context."""
    compact: dict[str, Any] = {"file_count": profile.get("file_count"), "files": {}}
    for name, meta in profile.get("files", {}).items():
        compact["files"][name] = {
            "row_count": meta.get("row_count"),
            "columns": meta.get("columns"),
            "sample_rows": meta.get("sample_rows", [])[:10],
            "candidate_join_keys": meta.get("candidate_join_keys"),
        }
    return json.dumps(compact, indent=2)


def create_cleaning_plan_llm(
    profile: dict[str, Any],
    frames: dict[str, pl.DataFrame],
) -> tuple[CleaningPlan, Literal["llm", "heuristic"]]:
    """Request a CleaningPlan from Azure OpenAI structured output."""
    settings = get_settings()
    if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
        return build_heuristic_cleaning_plan(profile, frames), "heuristic"

    client = get_client()
    deployment = get_deployment_name()
    system = (
        "You are a data engineering agent. Produce a JSON cleaning plan for messy operational "
        "CSVs/JSON files. Normalize column names to snake_case, strip currency symbols, parse "
        "mixed date formats, deduplicate on stable keys, and list required columns that must "
        "exist after cleaning. Prefer join_keys and dedup_keys from candidate_join_keys in the profile."
    )
    user = (
        "Dataset profile:\n"
        f"{_profile_prompt_payload(profile)}\n\n"
        "Return a CleaningPlan with: column_renames, dtype_fixes, dedup_keys, date_parsers, "
        "fill_strategies, currency_columns, required_columns, max_duplicate_rate (default 0.05), "
        "join_keys, primary_file."
    )

    try:
        completion = client.beta.chat.completions.parse(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=CleaningPlan,
            temperature=0,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is not None:
            return parsed, "llm"
    except Exception:
        pass

    try:
        schema = CleaningPlan.model_json_schema()
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system + f"\nJSON schema:\n{json.dumps(schema)}"},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        return CleaningPlan.model_validate_json(content), "llm"
    except (ValidationError, Exception):
        return build_heuristic_cleaning_plan(profile, frames), "heuristic"


def run_cleaning(
    profile_path: Path,
    artifact_dir: Path,
    *,
    plan: CleaningPlan | None = None,
    use_llm: bool = True,
    plan_source: Literal["llm", "heuristic", "provided"] | None = None,
) -> Path:
    """
    Run Phase 1 cleaning.

    Reads profile.json and raw tables, produces cleaned.parquet and cleaning_report.json.
    """
    profile_path = profile_path.resolve()
    artifact_dir = artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    profile = _load_profile(profile_path)
    paths = _paths_from_profile(profile)
    frames = load_files(paths)

    if plan is None:
        if use_llm:
            plan, source = create_cleaning_plan_llm(profile, frames)
        else:
            plan = build_heuristic_cleaning_plan(profile, frames)
            source = "heuristic"
    else:
        source = plan_source or "provided"

    cleaned_df, rows_before = execute_cleaning_plan(frames, plan)
    validation = validate_cleaned(cleaned_df, plan, rows_before)

    parquet_path = artifact_dir / "cleaned.parquet"
    report_path = artifact_dir / "cleaning_report.json"

    cleaned_df.write_parquet(parquet_path)

    report = CleaningReport(
        status="passed" if validation.passed else "failed",
        plan_source=source,
        input_files=[p.name for p in paths],
        validation=validation,
        plan=plan,
    )
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    if not validation.passed:
        raise CleaningValidationError(validation.message)

    return parquet_path
