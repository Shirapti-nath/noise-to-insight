"""Pydantic models for Phase 4 predictive analytics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    """Single point on the history or forecast curve."""

    date: str
    value: float
    lower: float | None = None
    upper: float | None = None


class PrescriptiveAction(BaseModel):
    """Recommended action with a target date."""

    action: str
    due_date: str = Field(description="ISO date YYYY-MM-DD")


class ForecastNarrative(BaseModel):
    """LLM structured output for forecast explanation."""

    forecast_narrative: str
    prescriptive_actions: list[PrescriptiveAction] = Field(min_length=1, max_length=3)


class ForecastResult(BaseModel):
    """Written to forecast.json."""

    status: Literal["success", "skipped", "failed", "snapshot"]
    user_message: str | None = None
    time_column: str | None = None
    target_column: str | None = None
    model: str | None = None
    horizon_days: int = 30
    history: list[ForecastPoint] = Field(default_factory=list)
    forecast: list[ForecastPoint] = Field(default_factory=list)
    forecast_narrative: str | None = None
    prescriptive_actions: list[PrescriptiveAction] = Field(default_factory=list)
    chart_path: str | None = None
    narrative_source: Literal["llm", "heuristic", "provided"] = "heuristic"
