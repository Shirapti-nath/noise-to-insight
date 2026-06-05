"""LangGraph state machine for the six-phase pipeline."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from src.config import ARTIFACTS_DIR, ensure_artifact_dir
from src.ingest.loader import load_files
from src.ingest.profiler import profile_dataset
from src.models.state import PhaseLogEntry, PipelineState
from src.phases.anomalies import run_anomalies
from src.phases.cleaning import run_cleaning
from src.phases.forecast import run_forecast
from src.phases.graph_build import run_graph_build
from src.phases.patterns import get_top_insights, run_patterns
from src.phases.report import run_report

PHASE_ORDER = (
    "ingest",
    "clean",
    "patterns",
    "anomalies",
    "forecast",
    "graph",
    "report",
)


def _paths_from_state(data: dict[str, Any]) -> PipelineState:
    """Deserialize graph state dict into PipelineState."""
    path_fields = (
        "artifact_dir",
        "profile_path",
        "cleaned_path",
        "patterns_path",
        "anomalies_path",
        "forecast_path",
        "graph_path",
        "report_path",
    )
    converted = dict(data)
    for key in path_fields:
        if converted.get(key):
            converted[key] = Path(converted[key])
    if converted.get("input_paths"):
        converted["input_paths"] = [Path(p) for p in converted["input_paths"]]
    if converted.get("phase_log"):
        converted["phase_log"] = [
            PhaseLogEntry.model_validate(e) if isinstance(e, dict) else e
            for e in converted["phase_log"]
        ]
    return PipelineState.model_validate(converted)


def _state_to_dict(state: PipelineState) -> dict[str, Any]:
    return json.loads(state.model_dump_json())


def _log_phase(state: PipelineState, phase: str, status: str, message: str, started: float) -> None:
    state.phase_log.append(
        PhaseLogEntry(
            phase=phase,
            status=status,
            message=message,
            duration_sec=round(time.perf_counter() - started, 2),
        )
    )


def _run_phase(
    state: PipelineState,
    phase: str,
    fn,
) -> PipelineState:
    """Execute a phase function with timing and error capture."""
    state.current_phase = phase
    started = time.perf_counter()
    try:
        fn(state)
        _log_phase(state, phase, "completed", "OK", started)
    except Exception as exc:
        state.error = f"{phase}: {exc}"
        _log_phase(state, phase, "failed", str(exc), started)
    return state


def node_ingest(state: dict[str, Any]) -> dict[str, Any]:
    s = _paths_from_state(state)

    def _work(st: PipelineState) -> None:
        raw_dir = st.artifact_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for path in st.input_paths:
            dest = raw_dir / path.name
            if path.resolve() != dest.resolve():
                shutil.copy2(path, dest)
        profile_path = st.artifact_dir / "profile.json"
        profile_dataset(st.input_paths, profile_path)
        st.profile_path = profile_path

    result = _run_phase(s, "ingest", _work)
    return _state_to_dict(result)


def node_clean(state: dict[str, Any]) -> dict[str, Any]:
    s = _paths_from_state(state)

    def _work(st: PipelineState) -> None:
        if not st.profile_path:
            raise ValueError("profile_path missing — run ingest first")
        st.cleaned_path = run_cleaning(
            st.profile_path,
            st.artifact_dir,
            use_llm=st.use_llm,
        )

    result = _run_phase(s, "clean", _work)
    return _state_to_dict(result)


def node_patterns(state: dict[str, Any]) -> dict[str, Any]:
    s = _paths_from_state(state)

    def _work(st: PipelineState) -> None:
        if not st.cleaned_path:
            raise ValueError("cleaned_path missing")
        st.patterns_path = run_patterns(st.cleaned_path, st.artifact_dir, use_llm=st.use_llm)
        top = get_top_insights(st.patterns_path, n=1)
        if top:
            st.headline_insight = f"{top[0].title} — {top[0].summary}"

    result = _run_phase(s, "patterns", _work)
    return _state_to_dict(result)


def node_anomalies(state: dict[str, Any]) -> dict[str, Any]:
    s = _paths_from_state(state)

    def _work(st: PipelineState) -> None:
        if not st.cleaned_path:
            raise ValueError("cleaned_path missing")
        st.anomalies_path = run_anomalies(st.cleaned_path, st.artifact_dir, use_llm=st.use_llm)

    result = _run_phase(s, "anomalies", _work)
    return _state_to_dict(result)


def node_forecast(state: dict[str, Any]) -> dict[str, Any]:
    s = _paths_from_state(state)

    def _work(st: PipelineState) -> None:
        if not st.cleaned_path:
            raise ValueError("cleaned_path missing")
        st.forecast_path = run_forecast(st.cleaned_path, st.artifact_dir, use_llm=st.use_llm)

    result = _run_phase(s, "forecast", _work)
    return _state_to_dict(result)


def node_graph(state: dict[str, Any]) -> dict[str, Any]:
    s = _paths_from_state(state)

    def _work(st: PipelineState) -> None:
        if not st.cleaned_path:
            raise ValueError("cleaned_path missing")
        highlight = None
        if st.anomalies_path and st.anomalies_path.exists():
            from src.phases.anomalies import load_anomalies

            anomalies = load_anomalies(st.anomalies_path)
            if anomalies.anomalies and anomalies.anomalies[0].graph_entity_id:
                highlight = anomalies.anomalies[0].graph_entity_id
        st.graph_path = run_graph_build(
            st.cleaned_path,
            st.artifact_dir,
            patterns_path=st.patterns_path,
            anomalies_path=st.anomalies_path,
            use_llm=st.use_llm,
            highlight_entity_id=highlight,
        )

    result = _run_phase(s, "graph", _work)
    return _state_to_dict(result)


def node_report(state: dict[str, Any]) -> dict[str, Any]:
    s = _paths_from_state(state)

    def _work(st: PipelineState) -> None:
        st.report_path = run_report(st.artifact_dir, use_llm=st.use_llm)

    result = _run_phase(s, "report", _work)
    return _state_to_dict(result)


def _route_on_error(state: dict[str, Any]) -> Literal["continue", "stop"]:
    if state.get("error"):
        return "stop"
    return "continue"


def build_pipeline_graph():
    """Compile the LangGraph pipeline."""
    workflow = StateGraph(dict)
    workflow.add_node("ingest", node_ingest)
    workflow.add_node("clean", node_clean)
    workflow.add_node("patterns", node_patterns)
    workflow.add_node("anomalies", node_anomalies)
    workflow.add_node("forecast", node_forecast)
    workflow.add_node("graph", node_graph)
    workflow.add_node("report", node_report)

    workflow.set_entry_point("ingest")
    for prev, nxt in zip(PHASE_ORDER, PHASE_ORDER[1:], strict=False):
        workflow.add_conditional_edges(
            prev,
            _route_on_error,
            {"continue": nxt, "stop": END},
        )
    workflow.add_conditional_edges("report", _route_on_error, {"continue": END, "stop": END})
    return workflow.compile()


def run_pipeline(
    input_paths: list[Path],
    run_id: str,
    *,
    use_llm: bool = True,
) -> PipelineState:
    """
    Run the full pipeline and return final PipelineState.

    Raises if any phase fails after recording error in state.
    """
    artifact_dir = ensure_artifact_dir(run_id)
    initial = PipelineState(
        run_id=run_id,
        artifact_dir=artifact_dir,
        input_paths=[Path(p).resolve() for p in input_paths],
        use_llm=use_llm,
    )

    graph = build_pipeline_graph()
    final_dict = graph.invoke(_state_to_dict(initial))
    final = _paths_from_state(final_dict)

    if final.error:
        raise RuntimeError(final.error)
    return final


def replay_golden_run(run_id: str = "golden_replay") -> Path:
    """Copy precomputed golden artifacts for offline demo."""
    golden = ARTIFACTS_DIR / "golden"
    if not golden.exists():
        raise FileNotFoundError(
            f"Golden run not found at {golden}. Run the pipeline once and copy artifacts there.",
        )
    dest = ensure_artifact_dir(run_id)
    shutil.copytree(golden, dest, dirs_exist_ok=True)
    return dest
