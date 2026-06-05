"""Pipeline state and shared data contracts."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class InsightCard(BaseModel):
    """Ranked business insight from pattern discovery."""

    title: str
    summary: str
    impact: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class AnomalyRecord(BaseModel):
    """Detected anomaly with optional LLM explanation."""

    entity: str
    metric: str
    score: float
    severity: str = "medium"
    hypothesis: str | None = None
    recommended_action: str | None = None
    graph_entity_id: str | None = None
    source: str = "isolation_forest"
    evidence: dict[str, Any] = Field(default_factory=dict)


class GraphPayload(BaseModel):
    """Knowledge graph nodes and edges for viz and report."""

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)


class PhaseLogEntry(BaseModel):
    """Timing and status for one pipeline phase."""

    phase: str
    status: str
    message: str = ""
    duration_sec: float = 0.0


class PipelineState(BaseModel):
    """Mutable state passed through LangGraph nodes."""

    run_id: str
    artifact_dir: Path
    input_paths: list[Path] = Field(default_factory=list)
    use_llm: bool = True
    profile_path: Path | None = None
    cleaned_path: Path | None = None
    patterns_path: Path | None = None
    anomalies_path: Path | None = None
    forecast_path: Path | None = None
    graph_path: Path | None = None
    report_path: Path | None = None
    current_phase: str = ""
    error: str | None = None
    phase_log: list[PhaseLogEntry] = Field(default_factory=list)
    headline_insight: str | None = None
