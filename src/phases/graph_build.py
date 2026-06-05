"""Phase 5: Knowledge graph construction from entities and relations."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import polars as pl
from pydantic import ValidationError

from src.config import get_settings
from src.llm.client import get_client, get_deployment_name
from src.models.anomalies import AnomalyDetectionResult
from src.models.graph import GraphBuildResult, GraphEdge, GraphExtraction, GraphNode
from src.models.patterns import PatternDiscoveryResult
from src.models.state import GraphPayload
from src.phases.anomalies import build_graph_entity_id
from src.viz.graph_viz import highlight_neighborhood, render_graph_html

ENTITY_COLUMN_PATTERN = re.compile(
    r"(^id$|_id$|.*_id$|^sku$|^region$|^warehouse|^supplier|^product|^customer|"
    r"department|jobrole|education|attrition|business|field|marital|gender|overtime|"
    r"joblevel|stock|involvement|satisfaction|environment|relationship|commute|"
    r"training|role|distance|frequency)",
    re.IGNORECASE,
)
MAX_LLM_SAMPLE_ROWS = 25
MAX_GRAPH_UNIQUE = 40
MAX_GRAPH_ROWS = 800


def _normalize_col_key(name: str) -> str:
    return re.sub(r"[\s-]+", "_", name.strip().lower())


def _fallback_graph_columns(df: pl.DataFrame) -> list[str]:
    """Categorical columns for graph when name patterns do not match."""
    cols: list[str] = []
    for col in df.columns:
        dtype = df[col].dtype
        if dtype not in (pl.Utf8, pl.Categorical, pl.String):
            continue
        n_unique = df[col].n_unique()
        if 2 <= n_unique <= MAX_GRAPH_UNIQUE:
            cols.append(col)
    return cols[:12]


def _graph_entity_columns(df: pl.DataFrame) -> list[str]:
    """Entity columns suitable for graph (skip ultra-high cardinality IDs)."""
    cols: list[str] = []
    for col in df.columns:
        normalized = _normalize_col_key(col)
        if not ENTITY_COLUMN_PATTERN.search(normalized):
            continue
        n_unique = df[col].n_unique()
        if 2 <= n_unique <= MAX_GRAPH_UNIQUE:
            cols.append(col)
    if not cols:
        cols = _fallback_graph_columns(df)
    return cols


def _node_label(node_id: str) -> str:
    if ":" in node_id:
        return node_id.split(":", 1)[1]
    return node_id


def _node_type_from_id(node_id: str) -> str:
    if ":" in node_id:
        return node_id.split(":", 1)[0]
    return "entity"


def extract_entities_heuristic(df: pl.DataFrame) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Build nodes and co-occurrence edges from tabular entity columns."""
    if df.height > MAX_GRAPH_ROWS:
        df = df.sample(MAX_GRAPH_ROWS, seed=42)
    entity_cols = _graph_entity_columns(df)
    if not entity_cols:
        return [], []

    node_map: dict[str, GraphNode] = {}
    edge_weights: dict[tuple[str, str, str], float] = {}

    for row in df.iter_rows(named=True):
        row_nodes: list[str] = []
        for col in entity_cols:
            val = row.get(col)
            if val is None or str(val).strip() == "":
                continue
            node_id = build_graph_entity_id(col, str(val))
            row_nodes.append(node_id)
            if node_id not in node_map:
                node_map[node_id] = GraphNode(
                    id=node_id,
                    label=_node_label(node_id),
                    node_type=_node_type_from_id(node_id),
                )

        for i, src in enumerate(row_nodes):
            for tgt in row_nodes[i + 1 :]:
                a, b = (src, tgt) if src < tgt else (tgt, src)
                relation = f"{_node_type_from_id(a)}_{_node_type_from_id(b)}"
                key = (a, b, relation)
                edge_weights[key] = edge_weights.get(key, 0.0) + 1.0

    edges = [
        GraphEdge(source=a, target=b, relation=rel, weight=w)
        for (a, b, rel), w in edge_weights.items()
    ]
    return list(node_map.values()), edges


def _context_for_llm(
    df: pl.DataFrame,
    patterns: PatternDiscoveryResult | None,
    anomalies: AnomalyDetectionResult | None,
) -> dict[str, Any]:
    sample = df.head(MAX_LLM_SAMPLE_ROWS).to_dicts()
    ctx: dict[str, Any] = {
        "columns": df.columns,
        "sample_rows": sample,
    }
    if patterns:
        ctx["top_insights"] = [i.model_dump() for i in patterns.insights[:3]]
    if anomalies:
        ctx["top_anomalies"] = [a.model_dump() for a in anomalies.anomalies[:5]]
    return ctx


def extract_entities_llm(
    df: pl.DataFrame,
    patterns: PatternDiscoveryResult | None = None,
    anomalies: AnomalyDetectionResult | None = None,
) -> tuple[list[GraphNode], list[GraphEdge], Literal["llm", "heuristic"]]:
    """LLM entity/relation extraction from samples and prior phase artifacts."""
    settings = get_settings()
    if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
        nodes, edges = extract_entities_heuristic(df)
        return nodes, edges, "heuristic"

    context = _context_for_llm(df, patterns, anomalies)
    system = (
        "Extract a knowledge graph from operational data. Return nodes with stable ids "
        "like 'supplier:SUP-9', 'sku:WIDGET-A', 'region:East' and edges with relation "
        "labels (supplies, ships_to, sold_in, delayed_by, etc.). Focus on relationships "
        "that explain anomalies and insights when provided."
    )

    client = get_client()
    deployment = get_deployment_name()

    try:
        completion = client.beta.chat.completions.parse(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(context, default=str)},
            ],
            response_format=GraphExtraction,
            temperature=0.1,
        )
        parsed = completion.choices[0].message.parsed
        if parsed and (parsed.nodes or parsed.edges):
            return parsed.nodes, parsed.edges, "llm"
    except Exception:
        pass

    try:
        schema = GraphExtraction.model_json_schema()
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system + f"\nSchema:\n{json.dumps(schema)}"},
                {"role": "user", "content": json.dumps(context, default=str)},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        parsed = GraphExtraction.model_validate_json(response.choices[0].message.content or "{}")
        if parsed.nodes or parsed.edges:
            return parsed.nodes, parsed.edges, "llm"
    except (ValidationError, Exception):
        pass

    nodes, edges = extract_entities_heuristic(df)
    return nodes, edges, "heuristic"


def merge_graph_elements(
    heuristic_nodes: list[GraphNode],
    heuristic_edges: list[GraphEdge],
    llm_nodes: list[GraphNode],
    llm_edges: list[GraphEdge],
) -> tuple[list[GraphNode], list[GraphEdge], Literal["llm", "heuristic", "hybrid"]]:
    """Merge heuristic and LLM graph elements."""
    node_map = {n.id: n for n in heuristic_nodes}
    for node in llm_nodes:
        if node.id not in node_map:
            node_map[node.id] = node
        else:
            existing = node_map[node.id]
            node_map[node.id] = existing.model_copy(
                update={
                    "label": node.label or existing.label,
                    "node_type": node.node_type or existing.node_type,
                    "metadata": {**existing.metadata, **node.metadata},
                },
            )

    edge_map: dict[tuple[str, str, str], GraphEdge] = {}
    for edge in heuristic_edges + llm_edges:
        a, b = (edge.source, edge.target) if edge.source < edge.target else (edge.target, edge.source)
        key = (a, b, edge.relation)
        if key in edge_map:
            edge_map[key] = edge_map[key].model_copy(
                update={"weight": edge_map[key].weight + edge.weight},
            )
        else:
            edge_map[key] = edge.model_copy(update={"source": a, "target": b})

    source: Literal["llm", "heuristic", "hybrid"] = "hybrid"
    if llm_nodes or llm_edges:
        source = "hybrid" if heuristic_nodes else "llm"
    else:
        source = "heuristic"

    return list(node_map.values()), list(edge_map.values()), source


def apply_anomaly_flags(
    nodes: list[GraphNode],
    anomalies: AnomalyDetectionResult | None,
) -> list[GraphNode]:
    """Mark nodes referenced by Phase 3 anomalies."""
    if not anomalies:
        return nodes

    flagged_ids = {
        a.graph_entity_id for a in anomalies.anomalies if a.graph_entity_id
    }
    updated: list[GraphNode] = []
    for node in nodes:
        if node.id in flagged_ids:
            updated.append(
                node.model_copy(
                    update={
                        "is_anomaly": True,
                        "metadata": {**node.metadata, "anomaly": True},
                    },
                ),
            )
        else:
            updated.append(node)
    return updated


def prune_graph_for_viz(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    anomalies: AnomalyDetectionResult | None,
    *,
    max_nodes: int = 60,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Keep the most important nodes so PyVis/Streamlit can render reliably."""
    if len(nodes) <= max_nodes:
        return nodes, edges

    must_keep = {n.id for n in nodes if n.is_anomaly}
    if anomalies:
        must_keep |= {a.graph_entity_id for a in anomalies.anomalies if a.graph_entity_id}

    ranked = sorted(nodes, key=lambda n: n.centrality, reverse=True)
    keep_ids = set(must_keep)
    for node in ranked:
        if len(keep_ids) >= max_nodes:
            break
        keep_ids.add(node.id)

    pruned_nodes = [n for n in nodes if n.id in keep_ids]
    pruned_edges = [e for e in edges if e.source in keep_ids and e.target in keep_ids]
    return pruned_nodes, pruned_edges


def compute_centrality(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> tuple[list[GraphNode], list[str]]:
    """Attach degree centrality and return top hub entity ids."""
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.id)
    for edge in edges:
        graph.add_edge(edge.source, edge.target, relation=edge.relation, weight=edge.weight)

    if graph.number_of_nodes() == 0:
        return nodes, []

    centrality = nx.degree_centrality(graph)
    updated = [
        node.model_copy(update={"centrality": round(centrality.get(node.id, 0.0), 4)})
        for node in nodes
    ]
    hubs = sorted(centrality, key=centrality.get, reverse=True)[:5]
    return updated, hubs


def build_knowledge_graph(
    df: pl.DataFrame,
    *,
    patterns: PatternDiscoveryResult | None = None,
    anomalies: AnomalyDetectionResult | None = None,
    use_llm: bool = True,
) -> GraphBuildResult:
    """Construct the full knowledge graph."""
    h_nodes, h_edges = extract_entities_heuristic(df)

    if use_llm:
        llm_nodes, llm_edges, _ = extract_entities_llm(df, patterns, anomalies)
        nodes, edges, source = merge_graph_elements(h_nodes, h_edges, llm_nodes, llm_edges)
    else:
        nodes, edges, source = h_nodes, h_edges, "heuristic"

    nodes = apply_anomaly_flags(nodes, anomalies)
    nodes, edges = prune_graph_for_viz(nodes, edges, anomalies)
    nodes, hubs = compute_centrality(nodes, edges)

    return GraphBuildResult(
        node_count=len(nodes),
        edge_count=len(edges),
        nodes=nodes,
        edges=edges,
        hub_entities=hubs,
        extraction_source=source,
    )


def run_graph_build(
    cleaned_path: Path,
    artifact_dir: Path,
    *,
    patterns_path: Path | None = None,
    anomalies_path: Path | None = None,
    use_llm: bool = True,
    highlight_entity_id: str | None = None,
) -> Path:
    """Build knowledge graph; returns path to graph.json."""
    cleaned_path = cleaned_path.resolve()
    artifact_dir = artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(cleaned_path)

    patterns = None
    if patterns_path and patterns_path.exists():
        from src.phases.patterns import load_patterns

        patterns = load_patterns(patterns_path)

    anomalies = None
    if anomalies_path and anomalies_path.exists():
        from src.phases.anomalies import load_anomalies

        anomalies = load_anomalies(anomalies_path)
        if highlight_entity_id is None and anomalies.anomalies:
            highlight_entity_id = anomalies.anomalies[0].graph_entity_id

    result = build_knowledge_graph(
        df,
        patterns=patterns,
        anomalies=anomalies,
        use_llm=use_llm,
    )

    json_path = artifact_dir / "graph.json"
    html_path = artifact_dir / "graph.html"

    highlight_ids = highlight_neighborhood(
        result.nodes,
        result.edges,
        highlight_entity_id,
    ) if highlight_entity_id else set()

    render_graph_html(
        result.nodes,
        result.edges,
        html_path,
        highlight_ids=highlight_ids,
    )
    from src.viz.graph_viz import render_graph_png

    render_graph_png(
        result.nodes,
        result.edges,
        artifact_dir / "graph.png",
        highlight_ids=highlight_ids,
    )

    payload = result.model_dump(mode="json")
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["html_path"] = html_path.name
    payload["highlight_entity_id"] = highlight_entity_id
    payload["highlight_nodes"] = sorted(highlight_ids)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


def load_graph(path: Path) -> GraphBuildResult:
    """Load graph.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("generated_at", "highlight_entity_id", "highlight_nodes"):
        data.pop(key, None)
    return GraphBuildResult.model_validate(data)


def highlight_path(
    graph_path: Path,
    entity_id: str,
    *,
    hops: int = 2,
) -> set[str]:
    """
    Return node ids on the neighborhood path around entity_id for demo storytelling.

    Used by Streamlit to pulse the supplier → SKU → region chain.
    """
    result = load_graph(graph_path)
    return highlight_neighborhood(result.nodes, result.edges, entity_id, hops=hops)
