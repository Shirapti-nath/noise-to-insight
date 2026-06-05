"""Regenerate forecast and graph visuals from cleaned.parquet (fixes stale runs)."""

from __future__ import annotations

import json
from pathlib import Path

from src.phases.forecast import run_forecast
from src.phases.graph_build import load_graph, run_graph_build


def _forecast_is_stale(artifact_dir: Path) -> bool:
    forecast_path = artifact_dir / "forecast.json"
    if not (artifact_dir / "cleaned.parquet").exists():
        return False
    if not forecast_path.exists():
        return True
    data = json.loads(forecast_path.read_text(encoding="utf-8"))
    status = data.get("status")
    time_col = (data.get("time_column") or "").lower()
    if time_col in ("overtime", "trainingtimeslastyear"):
        return True
    if status == "skipped" and not data.get("forecast"):
        return True
    if status in ("success", "snapshot") and not (artifact_dir / "forecast.png").exists():
        return True
    return False


def _graph_is_stale(artifact_dir: Path) -> bool:
    if not (artifact_dir / "cleaned.parquet").exists():
        return False
    graph_path = artifact_dir / "graph.json"
    if not graph_path.exists():
        return True
    if not (artifact_dir / "graph.png").exists():
        return True
    try:
        result = load_graph(graph_path)
        return result.node_count == 0
    except Exception:
        return True


def refresh_visual_artifacts(
    artifact_dir: Path,
    *,
    use_llm: bool = False,
) -> dict[str, str]:
    """
    Rebuild forecast.json/png and graph.json/html/png when missing or from an old pipeline.

    Returns short status messages for UI.
    """
    artifact_dir = artifact_dir.resolve()
    cleaned = artifact_dir / "cleaned.parquet"
    messages: dict[str, str] = {}

    if not cleaned.exists():
        messages["error"] = "No cleaned.parquet — run the full pipeline first."
        return messages

    anomalies = artifact_dir / "anomalies.json"
    patterns = artifact_dir / "patterns.json"

    if _forecast_is_stale(artifact_dir):
        try:
            run_forecast(cleaned, artifact_dir, use_llm=use_llm)
            messages["forecast"] = "Forecast chart regenerated."
        except Exception as exc:
            messages["forecast"] = f"Forecast refresh failed: {exc}"

    if _graph_is_stale(artifact_dir):
        try:
            run_graph_build(
                cleaned,
                artifact_dir,
                anomalies_path=anomalies if anomalies.exists() else None,
                patterns_path=patterns if patterns.exists() else None,
                use_llm=use_llm,
            )
            messages["graph"] = "Knowledge graph regenerated."
        except Exception as exc:
            messages["graph"] = f"Graph refresh failed: {exc}"

    return messages
