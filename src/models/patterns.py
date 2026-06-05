"""Pydantic models for Phase 2 pattern discovery."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.models.state import InsightCard


class RankedInsights(BaseModel):
    """LLM structured response: ranked insight cards."""

    insights: list[InsightCard] = Field(
        min_length=1,
        max_length=5,
        description="Top 3-5 business insights ranked by impact",
    )


class PatternDiscoveryResult(BaseModel):
    """Written to patterns.json."""

    row_count: int
    statistics: dict[str, Any] = Field(default_factory=dict)
    raw_patterns: list[dict[str, Any]] = Field(default_factory=list)
    insights: list[InsightCard] = Field(default_factory=list)
    insight_source: Literal["llm", "heuristic", "provided"] = "heuristic"
