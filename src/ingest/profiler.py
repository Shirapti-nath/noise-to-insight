"""Profile raw datasets for LLM context and cleaning plans."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl

from src.ingest.loader import load_files

JOIN_KEY_PATTERN = re.compile(
    r"(^id$|_id$|.*_id$|.*_key$|^key$|.*_code$|^code$|^sku$|.*_ref$|^ref$|.*_number$)",
    re.IGNORECASE,
)
SAMPLE_ROW_LIMIT = 20


def _json_safe(value: Any) -> Any:
    """Convert Polars/numpy values to JSON-serializable primitives."""
    if value is None:
        return None
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float) and (value != value):  # NaN
        return None
    return value


def _sample_rows(df: pl.DataFrame, limit: int = SAMPLE_ROW_LIMIT) -> list[dict[str, Any]]:
    if df.height == 0:
        return []
    sample = df.head(limit)
    rows: list[dict[str, Any]] = []
    for record in sample.to_dicts():
        rows.append({k: _json_safe(v) for k, v in record.items()})
    return rows


def _column_stats(df: pl.DataFrame) -> dict[str, dict[str, Any]]:
    n = df.height or 1
    stats: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        series = df[col]
        null_count = series.null_count()
        stats[col] = {
            "dtype": str(series.dtype),
            "null_count": null_count,
            "null_pct": round(100.0 * null_count / n, 2),
            "n_unique": series.n_unique(),
        }
    return stats


def candidate_join_keys(df: pl.DataFrame) -> list[str]:
    """Heuristic join-key detection from column names and uniqueness."""
    if df.height == 0:
        return []

    scored: list[tuple[int, str]] = []
    n = df.height

    for col in df.columns:
        null_pct = 100.0 * df[col].null_count() / n
        if null_pct > 50.0:
            continue

        n_unique = df[col].n_unique()
        uniq_ratio = n_unique / n if n > 0 else 0.0
        normalized = col.strip().replace(" ", "_")
        name_match = bool(JOIN_KEY_PATTERN.match(normalized))

        if name_match:
            scored.append((0, col))
        elif uniq_ratio >= 0.95 and n_unique > 1:
            scored.append((2, col))
        elif uniq_ratio >= 0.85 and n_unique > 1 and "id" in col.lower():
            scored.append((1, col))

    scored.sort(key=lambda x: (x[0], x[1]))
    seen: set[str] = set()
    result: list[str] = []
    for _, name in scored:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def build_profile(
    paths: list[Path],
    frames: dict[str, pl.DataFrame] | None = None,
) -> dict[str, Any]:
    """Build a profile dict for all input files."""
    if frames is None:
        frames = load_files(paths)

    path_by_name = {p.resolve().name: p.resolve() for p in paths}
    files_profile: dict[str, Any] = {}

    for name, df in frames.items():
        files_profile[name] = {
            "path": str(path_by_name.get(name, name)),
            "row_count": df.height,
            "column_count": df.width,
            "columns": _column_stats(df),
            "sample_rows": _sample_rows(df),
            "candidate_join_keys": candidate_join_keys(df),
        }

    return {
        "file_count": len(files_profile),
        "files": files_profile,
    }


def profile_dataset(
    paths: list[Path],
    output_path: Path,
    frames: dict[str, pl.DataFrame] | None = None,
) -> Path:
    """Emit profile.json for the given inputs."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = build_profile(paths, frames=frames)
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return output_path
