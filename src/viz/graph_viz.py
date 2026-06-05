"""PyVis / NetworkX graph rendering."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from src.models.graph import GraphEdge, GraphNode

# Non-interactive backend for server/Streamlit
import matplotlib

matplotlib.use("Agg")

NODE_COLOR_DEFAULT = "#3b82f6"
NODE_COLOR_ANOMALY = "#dc2626"
NODE_COLOR_HIGHLIGHT = "#f59e0b"
NODE_COLOR_HUB = "#7c3aed"


def highlight_neighborhood(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    entity_id: str | None,
    *,
    hops: int = 2,
) -> set[str]:
    """Collect node ids within ``hops`` steps of ``entity_id``."""
    if not entity_id:
        return set()

    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.id)
    for edge in edges:
        graph.add_edge(edge.source, edge.target)

    if entity_id not in graph:
        return {entity_id}

    visited = {entity_id}
    frontier = {entity_id}
    for _ in range(hops):
        nxt: set[str] = set()
        for nid in frontier:
            nxt.update(graph.neighbors(nid))
        nxt -= visited
        visited |= nxt
        frontier = nxt
    return visited


def _node_color(node: GraphNode, highlight_ids: set[str], hub_ids: set[str]) -> str:
    if node.id in highlight_ids:
        return NODE_COLOR_HIGHLIGHT
    if node.is_anomaly:
        return NODE_COLOR_ANOMALY
    if node.id in hub_ids:
        return NODE_COLOR_HUB
    return NODE_COLOR_DEFAULT


def render_graph_html(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    output_path: Path,
    *,
    highlight_ids: set[str] | None = None,
    height: str = "600px",
    width: str = "100%",
) -> Path:
    """Render interactive graph HTML via PyVis (inline CDN for Streamlit iframe)."""
    from pyvis.network import Network

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    highlight_ids = highlight_ids or set()
    hub_ids = {n.id for n in sorted(nodes, key=lambda n: n.centrality, reverse=True)[:3]}

    if not nodes:
        output_path.write_text("<html><body><p>No graph nodes to display.</p></body></html>")
        return output_path

    net = Network(
        height=height,
        width=width,
        bgcolor="#ffffff",
        font_color="#1e293b",
        directed=False,
        cdn_resources="in_line",
    )
    net.barnes_hut(
        gravity=-2000,
        central_gravity=0.35,
        spring_length=95,
        spring_strength=0.05,
    )

    for node in nodes:
        size = 12 + (node.centrality * 30)
        if node.id in highlight_ids:
            size += 8
        title = (
            f"{node.label}\n"
            f"type: {node.node_type}\n"
            f"centrality: {node.centrality}\n"
            f"anomaly: {node.is_anomaly}"
        )
        net.add_node(
            node.id,
            label=node.label[:24],
            title=title,
            color=_node_color(node, highlight_ids, hub_ids),
            size=size,
            borderWidth=3 if node.id in highlight_ids else 1,
        )

    for edge in edges:
        net.add_edge(
            edge.source,
            edge.target,
            title=edge.relation,
            width=max(1, min(edge.weight, 4)),
        )

    net.save_graph(str(output_path))
    return output_path


def render_graph_png(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    output_path: Path,
    *,
    highlight_ids: set[str] | None = None,
) -> Path | None:
    """Static PNG fallback for Streamlit when iframe HTML is blank."""
    if not nodes:
        return None

    highlight_ids = highlight_ids or set()
    hub_ids = {n.id for n in sorted(nodes, key=lambda n: n.centrality, reverse=True)[:3]}

    graph = nx.Graph()
    color_map: dict[str, str] = {}
    size_map: dict[str, float] = {}
    label_map: dict[str, str] = {}
    for node in nodes:
        graph.add_node(node.id)
        color_map[node.id] = _node_color(node, highlight_ids, hub_ids)
        size_map[node.id] = 80 + node.centrality * 400
        label_map[node.id] = node.label
    for edge in edges:
        if edge.source in graph and edge.target in graph:
            graph.add_edge(edge.source, edge.target)

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(graph, seed=42, k=0.9)
    for node_id, (x, y) in pos.items():
        ax.scatter(
            x,
            y,
            s=size_map.get(node_id, 80),
            c=color_map.get(node_id, NODE_COLOR_DEFAULT),
            edgecolors="#1e293b",
            linewidths=0.5,
            zorder=2,
        )
        ax.text(x, y, label_map.get(node_id, node_id)[:16], fontsize=6, ha="center")
    for src, tgt in graph.edges:
        x1, y1 = pos[src]
        x2, y2 = pos[tgt]
        ax.plot([x1, x2], [y1, y2], color="#94a3b8", linewidth=0.6, zorder=1)

    ax.set_title("Knowledge graph (key entities & links)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def render_graph_figure(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    *,
    highlight_ids: set[str] | None = None,
):
    """Return a matplotlib figure for Streamlit ``st.pyplot`` (always visible)."""
    if not nodes:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No graph nodes", ha="center", va="center")
        ax.axis("off")
        return fig

    highlight_ids = highlight_ids or set()
    hub_ids = {n.id for n in sorted(nodes, key=lambda n: n.centrality, reverse=True)[:3]}

    graph = nx.Graph()
    color_map: dict[str, str] = {}
    size_map: dict[str, float] = {}
    label_map: dict[str, str] = {}
    for node in nodes:
        graph.add_node(node.id)
        color_map[node.id] = _node_color(node, highlight_ids, hub_ids)
        size_map[node.id] = 80 + node.centrality * 400
        label_map[node.id] = node.label
    for edge in edges:
        if edge.source in graph and edge.target in graph:
            graph.add_edge(edge.source, edge.target)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    pos = nx.spring_layout(graph, seed=42, k=0.9)
    for node_id, (x, y) in pos.items():
        ax.scatter(
            x,
            y,
            s=size_map.get(node_id, 80),
            c=color_map.get(node_id, NODE_COLOR_DEFAULT),
            edgecolors="#1e293b",
            linewidths=0.5,
            zorder=2,
        )
        ax.text(
            x,
            y,
            label_map.get(node_id, node_id)[:14],
            fontsize=7,
            ha="center",
            color="#0f172a",
        )
    for src, tgt in graph.edges:
        x1, y1 = pos[src]
        x2, y2 = pos[tgt]
        ax.plot([x1, x2], [y1, y2], color="#64748b", linewidth=0.7, zorder=1)

    ax.set_title("Knowledge graph — connected entities", color="#0f172a", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


def render_graph_html_from_json(
    graph_path: Path,
    output_path: Path | None = None,
    *,
    highlight_entity_id: str | None = None,
) -> Path:
    """Load graph.json and render HTML."""
    from src.phases.graph_build import load_graph

    result = load_graph(graph_path)
    out = output_path or graph_path.parent / "graph.html"
    highlight_ids = highlight_neighborhood(
        result.nodes,
        result.edges,
        highlight_entity_id,
    ) if highlight_entity_id else set()
    return render_graph_html(result.nodes, result.edges, out, highlight_ids=highlight_ids)
