"""Load raw CSV/JSON files into Polars DataFrames."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import polars as pl

SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".ndjson"}
ENCODINGS_TO_TRY = ("utf-8-sig", "utf-8", "latin-1", "cp1252")
DELIMITERS = (",", ";", "\t", "|")


def expand_input_paths(patterns: list[str]) -> list[Path]:
    """Expand glob patterns and collect existing file paths."""
    found: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            found.extend(Path(m).resolve() for m in matches)
            continue
        path = Path(pattern)
        if path.exists():
            found.append(path.resolve())
    unique = sorted({p for p in found if p.is_file()})
    if not unique:
        raise FileNotFoundError(f"No files matched: {patterns}")
    return unique


def detect_encoding(path: Path, sample_size: int = 100_000) -> str:
    """Detect text encoding by trial decode on a byte sample."""
    raw = path.read_bytes()[:sample_size]
    for encoding in ENCODINGS_TO_TRY:
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "utf-8"


def detect_delimiter(path: Path, encoding: str, sample_size: int = 8192) -> str:
    """Sniff CSV delimiter from a text sample."""
    try:
        text = path.read_bytes()[:sample_size].decode(encoding, errors="replace")
        if not text.strip():
            return ","
        dialect = csv.Sniffer().sniff(text, delimiters="".join(DELIMITERS))
        return dialect.delimiter
    except (csv.Error, UnicodeDecodeError):
        return ","


def _read_csv(path: Path) -> pl.DataFrame:
    encoding = detect_encoding(path)
    separator = detect_delimiter(path, encoding)
    return pl.read_csv(
        path,
        encoding=encoding,
        separator=separator,
        infer_schema_length=10_000,
        try_parse_dates=False,
        ignore_errors=False,
    )


def _read_json(path: Path) -> pl.DataFrame:
    """Load JSON as a Polars frame (array of objects, NDJSON, or single object)."""
    text = path.read_text(encoding=detect_encoding(path))
    stripped = text.strip()
    if not stripped:
        return pl.DataFrame()

    if stripped.startswith("["):
        records = json.loads(stripped)
        if not records:
            return pl.DataFrame()
        if isinstance(records, list) and isinstance(records[0], dict):
            return pl.from_dicts(records, infer_schema_length=10_000)
        raise ValueError(f"Unsupported JSON array shape in {path}")

    # NDJSON / JSONL
    if path.suffix.lower() in {".jsonl", ".ndjson"} or "\n" in stripped and stripped[0] == "{":
        try:
            return pl.read_ndjson(path)
        except Exception:
            lines = [json.loads(line) for line in stripped.splitlines() if line.strip()]
            return pl.from_dicts(lines, infer_schema_length=10_000) if lines else pl.DataFrame()

    payload = json.loads(stripped)
    if isinstance(payload, dict):
        for key in ("data", "records", "items", "events", "results"):
            if key in payload and isinstance(payload[key], list):
                return pl.from_dicts(payload[key], infer_schema_length=10_000)
        return pl.from_dicts([payload], infer_schema_length=10_000)
    raise ValueError(f"Unsupported JSON structure in {path}")


def load_file(path: Path) -> pl.DataFrame:
    """Load a single CSV or JSON file into a DataFrame."""
    path = path.resolve()
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}' for {path.name}. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    if suffix == ".csv":
        return _read_csv(path)
    return _read_json(path)


def load_files(paths: list[Path]) -> dict[str, pl.DataFrame]:
    """Load multiple data files; keys are file names."""
    if not paths:
        raise ValueError("At least one input path is required")
    frames: dict[str, pl.DataFrame] = {}
    for path in paths:
        resolved = path.resolve()
        frames[resolved.name] = load_file(resolved)
    return frames


def main(argv: list[str] | None = None) -> int:
    """CLI entry: load files and write profile.json."""
    from src.ingest.profiler import profile_dataset

    parser = argparse.ArgumentParser(
        description="Load messy CSV/JSON files and emit profile.json",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="File paths or globs (e.g. data/demo/*.csv)",
    )
    parser.add_argument(
        "--profile-out",
        type=Path,
        default=None,
        help="Output path for profile.json (default: data/artifacts/ingest/profile.json)",
    )
    args = parser.parse_args(argv)

    paths = expand_input_paths(args.input)
    frames = load_files(paths)

    if args.profile_out is None:
        from src.config import ARTIFACTS_DIR

        out = ARTIFACTS_DIR / "ingest" / "profile.json"
    else:
        out = args.profile_out

    profile_path = profile_dataset(paths, out, frames=frames)
    print(f"Loaded {len(frames)} file(s): {', '.join(frames.keys())}")
    print(f"Profile written to {profile_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
