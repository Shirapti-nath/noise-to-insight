"""Pydantic models for Phase 5 knowledge graph."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.models.state import GraphPayload


class GraphNode(BaseModel):
    """Knowledge graph node."""

    id: str
    label: str
    node_type: str = "entity"
    centrality: float = 0.0
    is_anomaly: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """Knowledge graph edge."""

    source: str
    target: str
    relation: str = "related_to"
    weight: float = 1.0


class GraphExtraction(BaseModel):
    """LLM structured output for entities and relations."""

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class GraphBuildResult(BaseModel):
    """Written to graph.json."""

    node_count: int = 0
    edge_count: int = 0
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    hub_entities: list[str] = Field(
        default_factory=list,
        description="Top nodes by degree centrality",
    )
    extraction_source: Literal["llm", "heuristic", "hybrid"] = "heuristic"
    html_path: str | None = None

    def to_payload(self) -> GraphPayload:
        return GraphPayload(
            nodes=[n.model_dump() for n in self.nodes],
            edges=[e.model_dump() for e in self.edges],
        )
