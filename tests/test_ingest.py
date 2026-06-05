"""Tests for ingestion loader and profiler."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from src.ingest.loader import (
    detect_delimiter,
    detect_encoding,
    expand_input_paths,
    load_file,
    load_files,
    main,
)
from src.ingest.profiler import build_profile, candidate_join_keys, profile_dataset

FIXTURES = Path(__file__).parent / "fixtures"


def test_detect_encoding_utf8() -> None:
    path = FIXTURES / "orders.csv"
    assert detect_encoding(path) in ("utf-8", "utf-8-sig")


def test_detect_delimiter_comma() -> None:
    path = FIXTURES / "orders.csv"
    enc = detect_encoding(path)
    assert detect_delimiter(path, enc) == ","


def test_detect_delimiter_semicolon() -> None:
    path = FIXTURES / "inventory_semicolon.csv"
    enc = detect_encoding(path)
    assert detect_delimiter(path, enc) == ";"


def test_load_csv_comma() -> None:
    df = load_file(FIXTURES / "orders.csv")
    assert df.height == 3
    assert "order_id" in df.columns
    assert df["amount"].null_count() == 1


def test_load_csv_semicolon() -> None:
    df = load_file(FIXTURES / "inventory_semicolon.csv")
    assert df.height == 3
    assert set(df.columns) == {"sku", "warehouse_id", "qty"}


def test_load_json_array() -> None:
    df = load_file(FIXTURES / "events.json")
    assert df.height == 3
    assert "supplier_id" in df.columns


def test_load_files_multiple() -> None:
    paths = [
        FIXTURES / "orders.csv",
        FIXTURES / "inventory_semicolon.csv",
        FIXTURES / "events.json",
    ]
    frames = load_files(paths)
    assert len(frames) == 3
    assert "orders.csv" in frames
    assert frames["orders.csv"].height == 3


def test_expand_input_paths_glob() -> None:
    pattern = str(FIXTURES / "*.csv")
    paths = expand_input_paths([pattern])
    names = {p.name for p in paths}
    assert "orders.csv" in names
    assert "inventory_semicolon.csv" in names


def test_candidate_join_keys() -> None:
    df = pl.DataFrame(
        {
            "order_id": ["A", "B", "C"],
            "sku": ["X", "X", "Y"],
            "note": ["n1", "n2", "n3"],
        }
    )
    keys = candidate_join_keys(df)
    assert "order_id" in keys
    assert "sku" in keys or "order_id" in keys


def test_build_profile_structure() -> None:
    paths = [FIXTURES / "orders.csv", FIXTURES / "events.json"]
    profile = build_profile(paths)

    assert profile["file_count"] == 2
    orders = profile["files"]["orders.csv"]
    assert orders["row_count"] == 3
    assert "order_id" in orders["columns"]
    assert orders["columns"]["order_id"]["null_pct"] == 0.0
    assert len(orders["sample_rows"]) <= 20
    assert "order_id" in orders["candidate_join_keys"]

    events = profile["files"]["events.json"]
    assert events["row_count"] == 3
    assert "event_id" in events["candidate_join_keys"]


def test_profile_dataset_writes_json(tmp_path: Path) -> None:
    paths = [FIXTURES / "orders.csv"]
    out = tmp_path / "profile.json"
    result = profile_dataset(paths, out)

    assert result == out
    payload = json.loads(out.read_text())
    assert payload["files"]["orders.csv"]["row_count"] == 3


def test_cli_main_writes_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "cli_profile.json"
    argv = [
        "--input",
        str(FIXTURES / "orders.csv"),
        str(FIXTURES / "events.json"),
        "--profile-out",
        str(out),
    ]
    assert main(argv) == 0
    data = json.loads(out.read_text())
    assert data["file_count"] == 2


def test_expand_input_paths_raises_when_missing() -> None:
    with pytest.raises(FileNotFoundError):
        expand_input_paths(["/nonexistent/path/*.csv"])
