"""Tests for Phase 5 knowledge graph."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from src.models.graph import GraphEdge, GraphNode
from src.phases.anomalies import discover_anomalies, run_anomalies
from src.phases.graph_build import (
    build_knowledge_graph,
    extract_entities_heuristic,
    highlight_path,
    load_graph,
    run_graph_build,
)
from src.viz.graph_viz import highlight_neighborhood, render_graph_html

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def segment_df() -> pl.DataFrame:
    return pl.read_csv(FIXTURES / "segment_orders.csv", try_parse_dates=False).with_columns(
        pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col("amount").cast(pl.Float64),
    )


def test_extract_entities_heuristic(segment_df: pl.DataFrame) -> None:
    nodes, edges = extract_entities_heuristic(segment_df)
    node_ids = {n.id for n in nodes}
    assert "region:East" in node_ids
    assert "sku:WIDGET-A" in node_ids
    assert any(e.source == "region:East" or e.target == "region:East" for e in edges)


def test_build_knowledge_graph_heuristic(segment_df: pl.DataFrame) -> None:
    result = build_knowledge_graph(segment_df, use_llm=False)
    assert result.node_count >= 4
    assert result.edge_count >= 3
    assert result.hub_entities
    assert result.extraction_source == "heuristic"


def test_highlight_neighborhood(segment_df: pl.DataFrame) -> None:
    nodes, edges = extract_entities_heuristic(segment_df)
    highlighted = highlight_neighborhood(nodes, edges, "region:East", hops=1)
    assert "region:East" in highlighted
    assert len(highlighted) >= 2


def test_run_graph_build_writes_artifacts(tmp_path: Path, segment_df: pl.DataFrame) -> None:
    cleaned_path = tmp_path / "cleaned.parquet"
    artifact_dir = tmp_path / "run"
    segment_df.write_parquet(cleaned_path)

    graph_path = run_graph_build(cleaned_path, artifact_dir, use_llm=False)
    assert graph_path.exists()
    assert (artifact_dir / "graph.html").exists()
    assert (artifact_dir / "graph.png").exists()

    payload = json.loads(graph_path.read_text())
    assert payload["node_count"] >= 4
    assert payload["html_path"] == "graph.html"
    assert "highlight_nodes" in payload


def test_graph_links_anomaly_entities(tmp_path: Path) -> None:
    anomaly_df = pl.read_csv(FIXTURES / "anomaly_orders.csv", try_parse_dates=False).with_columns(
        pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col("amount").cast(pl.Float64),
    )
    cleaned_path = tmp_path / "cleaned.parquet"
    artifact_dir = tmp_path / "run"
    anomaly_df.write_parquet(cleaned_path)

    run_anomalies(cleaned_path, artifact_dir, use_llm=False)
    graph_path = run_graph_build(
        cleaned_path,
        artifact_dir,
        anomalies_path=artifact_dir / "anomalies.json",
        use_llm=False,
    )

    result = load_graph(graph_path)
    anomaly_nodes = [n for n in result.nodes if n.is_anomaly]
    assert anomaly_nodes
    assert any(n.id == "order:ORD-9" for n in anomaly_nodes)


def test_highlight_path_helper(tmp_path: Path, segment_df: pl.DataFrame) -> None:
    cleaned_path = tmp_path / "cleaned.parquet"
    artifact_dir = tmp_path / "run"
    segment_df.write_parquet(cleaned_path)
    graph_path = run_graph_build(
        cleaned_path,
        artifact_dir,
        use_llm=False,
        highlight_entity_id="region:East",
    )

    nodes = highlight_path(graph_path, "region:East")
    assert "region:East" in nodes


def test_render_graph_html_file(tmp_path: Path, segment_df: pl.DataFrame) -> None:
    nodes, edges = extract_entities_heuristic(segment_df)
    out = tmp_path / "test.html"
    render_graph_html(nodes, edges, out, highlight_ids={"region:East"})
    assert out.exists()
    assert "region:East" in out.read_text()


@patch("src.phases.graph_build.extract_entities_llm")
def test_run_graph_build_hybrid(mock_llm, tmp_path: Path, segment_df: pl.DataFrame) -> None:
    mock_llm.return_value = (
        [
            GraphNode(id="supplier:APEX", label="APEX", node_type="supplier"),
        ],
        [GraphEdge(source="supplier:APEX", target="sku:WIDGET-A", relation="supplies")],
        "llm",
    )

    cleaned_path = tmp_path / "cleaned.parquet"
    artifact_dir = tmp_path / "run_llm"
    segment_df.write_parquet(cleaned_path)

    run_graph_build(cleaned_path, artifact_dir, use_llm=True)
    mock_llm.assert_called_once()

    result = load_graph(artifact_dir / "graph.json")
    assert result.extraction_source == "hybrid"
    assert any(n.id == "supplier:APEX" for n in result.nodes)
