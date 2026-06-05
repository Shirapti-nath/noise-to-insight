"""Pydantic models for Phase 6 executive report."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExecutiveReportContent(BaseModel):
    """LLM-generated executive report prose."""

    executive_summary: str = Field(
        description="CFO-ready executive summary, maximum 250 words",
    )
    top_actions: list[str] = Field(
        min_length=1,
        max_length=5,
        description="Prioritized action bullets for leadership",
    )
