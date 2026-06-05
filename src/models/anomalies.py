"""Pydantic models for Phase 3 anomaly detection."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.models.state import AnomalyRecord


class ExplainedAnomalies(BaseModel):
    """LLM structured output: enriched anomaly records."""

    anomalies: list[AnomalyRecord] = Field(min_length=1, max_length=20)


class AnomalyDetectionResult(BaseModel):
    """Written to anomalies.json."""

    row_count: int
    feature_columns: list[str] = Field(default_factory=list)
    anomalies: list[AnomalyRecord] = Field(default_factory=list)
    entity_index: dict[str, list[int]] = Field(
        default_factory=dict,
        description="graph_entity_id -> indices in anomalies list",
    )
    explanation_source: Literal["llm", "heuristic", "provided"] = "heuristic"
