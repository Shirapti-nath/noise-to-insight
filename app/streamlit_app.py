"""Streamlit mission control — full six-phase results dashboard."""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import polars as pl
import streamlit as st
import streamlit.components.v1 as components

from src.config import ARTIFACTS_DIR, DEMO_DIR, PROJECT_ROOT, get_settings
from src.orchestrator.graph import PHASE_ORDER, replay_golden_run, run_pipeline
from src.phases.anomalies import load_anomalies
from src.phases.forecast import load_forecast
from src.phases.graph_build import load_graph
from src.phases.patterns import get_top_insights, load_patterns
from src.phases.report import load_report_meta
from src.viz.graph_viz import render_graph_figure
from src.viz.refresh import refresh_visual_artifacts

REPORT_EMBED_CSS = """
<style>
  html, body {
    background: #ffffff !important;
    color: #0f172a !important;
  }
  table, th, td, p, h1, h2, li, div, span {
    color: #0f172a !important;
  }
  th { background: #f1f5f9 !important; }
  .summary { background: #eff6ff !important; }
  .impact-high { color: #b91c1c !important; }
  .impact-medium { color: #b45309 !important; }
</style>
"""


def _wrap_report_for_streamlit(html: str) -> str:
    """Force readable light theme inside Streamlit dark UI iframe."""
    if "<head>" in html:
        return html.replace("<head>", f"<head>{REPORT_EMBED_CSS}", 1)
    return REPORT_EMBED_CSS + html

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
GOLDEN_DIR = ARTIFACTS_DIR / "golden"

PHASE_LABELS = {
    "ingest": "0 · Ingest & Profile",
    "clean": "1 · Data Cleaning",
    "patterns": "2 · Pattern Discovery",
    "anomalies": "3 · Anomaly Detection",
    "forecast": "4 · Predictive Analytics",
    "graph": "5 · Knowledge Graph",
    "report": "6 · Executive Report",
}


def _azure_configured() -> bool:
    settings = get_settings()
    return bool(settings.azure_openai_api_key and settings.azure_openai_endpoint)


def _demo_file_paths() -> list[Path]:
    if DEMO_DIR.exists() and any(DEMO_DIR.iterdir()):
        return sorted(
            p for p in DEMO_DIR.iterdir() if p.suffix.lower() in {".csv", ".json", ".jsonl"}
        )
    return sorted(
        p
        for p in (
            FIXTURES_DIR / "segment_orders.csv",
            FIXTURES_DIR / "forecast_series.csv",
        )
        if p.exists()
    )


def _save_uploads(uploaded_files: list) -> tuple[list[Path], Path]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    run_dir = ARTIFACTS_DIR / run_id
    raw_dir = run_dir / "uploads"
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for uploaded in uploaded_files:
        dest = raw_dir / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        paths.append(dest)
    return paths, run_dir


def _phase_status(phase_log: list[dict], phase: str) -> str:
    for entry in reversed(phase_log):
        if entry.get("phase") == phase:
            return entry.get("status", "pending")
    return "pending"


def _render_stepper(phase_log: list[dict]) -> None:
    cols = st.columns(len(PHASE_ORDER))
    for col, phase in zip(cols, PHASE_ORDER, strict=True):
        status = _phase_status(phase_log, phase)
        icon = {"completed": "✅", "failed": "❌"}.get(status, "⏳")
        col.markdown(f"{icon} **{PHASE_LABELS[phase]}**")


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _kpi_row(artifact_dir: Path) -> None:
    profile = _load_json(artifact_dir / "profile.json") or {}
    cleaning = _load_json(artifact_dir / "cleaning_report.json") or {}
    patterns = _load_json(artifact_dir / "patterns.json") or {}
    anomalies = _load_json(artifact_dir / "anomalies.json") or {}
    graph = _load_json(artifact_dir / "graph.json") or {}
    forecast = _load_json(artifact_dir / "forecast.json") or {}

    total_rows = 0
    if isinstance(profile.get("files"), dict):
        for meta in profile["files"].values():
            total_rows += int(meta.get("row_count", 0))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows ingested", f"{total_rows:,}" if total_rows else "—")
    c2.metric(
        "Clean status",
        str(cleaning.get("status", "—")).upper(),
        delta=(str((cleaning.get("validation") or {}).get("message") or ""))[:40] or None,
    )
    c3.metric("Insights", len(patterns.get("insights", [])))
    c4.metric("Anomalies", len(anomalies.get("anomalies", [])))
    c5.metric(
        "Graph",
        f"{graph.get('node_count', 0)} nodes · {graph.get('edge_count', 0)} edges",
    )
    forecast_status = forecast.get("status") or "skipped"
    if forecast_status == "success":
        st.caption(
            f"Forecast: {forecast.get('target_column') or 'target'} · "
            f"{forecast.get('horizon_days', 30)}-day horizon"
        )
    elif forecast_status == "snapshot":
        st.caption(
            f"Forecast: snapshot · {forecast.get('target_column') or 'target'} "
            f"by {forecast.get('time_column') or 'group'}"
        )
    elif forecast:
        msg = (forecast.get("user_message") or forecast.get("forecast_narrative") or "")[:120]
        st.caption(f"Forecast: {forecast_status}" + (f" — {msg}" if msg else ""))


def _tab_overview(artifact_dir: Path) -> None:
    patterns_path = artifact_dir / "patterns.json"
    st.subheader("Executive headline")
    if patterns_path.exists():
        top = get_top_insights(patterns_path, n=1)
        if top:
            st.success(f"**{top[0].title}**")
            st.write(top[0].summary)
            if top[0].impact:
                st.info(f"Business impact: **{top[0].impact}**")
        patterns = load_patterns(patterns_path)
        if len(patterns.insights) > 1:
            with st.expander(f"All {len(patterns.insights)} insights", expanded=False):
                for ins in patterns.insights:
                    st.markdown(f"**{ins.title}** — {ins.summary}")
                    if ins.impact:
                        st.caption(f"Impact: {ins.impact}")
    else:
        st.warning("No pattern insights generated.")

    meta = load_report_meta(artifact_dir)
    if meta.get("executive_summary"):
        st.subheader("Report summary")
        st.write(meta["executive_summary"])


def _tab_ingest(artifact_dir: Path) -> None:
    profile = _load_json(artifact_dir / "profile.json")
    if not profile:
        st.warning("No profile.json — run ingest phase.")
        return

    st.markdown("**Data quality snapshot** after load and profiling.")
    files = profile.get("files", {})
    for name, meta in files.items():
        st.markdown(f"### `{name}`")
        st.write(
            f"**{meta.get('row_count', 0):,}** rows · "
            f"**{meta.get('column_count', 0)}** columns"
        )
        cols = meta.get("columns", {})
        if cols:
            rows = []
            for col_name, col_meta in cols.items():
                rows.append(
                    {
                        "column": col_name,
                        "dtype": col_meta.get("dtype"),
                        "null %": col_meta.get("null_pct"),
                        "unique": col_meta.get("n_unique"),
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Raw profile.json"):
        st.json(profile)


def _tab_cleaning(artifact_dir: Path) -> None:
    report = _load_json(artifact_dir / "cleaning_report.json")
    cleaned_path = artifact_dir / "cleaned.parquet"

    if not report:
        st.warning("No cleaning_report.json.")
        return

    st.markdown(f"**Status:** `{report.get('status')}` · **Plan source:** `{report.get('plan_source')}`")
    validation = report.get("validation", {})
    if validation:
        v1, v2, v3 = st.columns(3)
        v1.metric("Rows before", validation.get("rows_before", "—"))
        v2.metric("Rows after", validation.get("rows_after", "—"))
        v3.metric("Duplicate rate", f"{validation.get('duplicate_rate', 0):.2%}")

    plan = report.get("plan", {})
    if plan.get("column_renames"):
        st.markdown("**Column normalization**")
        st.dataframe(plan["column_renames"], use_container_width=True, hide_index=True)
    if plan.get("drops"):
        st.markdown("**Dropped columns**")
        st.write(", ".join(plan["drops"]))

    if cleaned_path.exists():
        st.markdown("**Cleaned preview** (first 25 rows)")
        df = pl.read_parquet(cleaned_path)
        st.dataframe(df.head(25).to_pandas(), use_container_width=True)

    with st.expander("Full cleaning_report.json"):
        st.json(report)


def _tab_patterns(artifact_dir: Path) -> None:
    path = artifact_dir / "patterns.json"
    if not path.exists():
        st.warning("No patterns.json.")
        return

    patterns = load_patterns(path)
    st.markdown(
        f"**{len(patterns.insights)}** ranked insights · "
        f"**{len(patterns.raw_patterns)}** raw patterns · "
        f"source: `{patterns.insight_source}`"
    )

    stats = patterns.statistics or {}
    if stats.get("pattern_counts"):
        st.json(stats["pattern_counts"])

    rows = []
    for ins in patterns.insights:
        rows.append(
            {
                "title": ins.title,
                "impact": ins.impact,
                "summary": ins.summary,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    corr_rows = [p for p in patterns.raw_patterns if p.get("type") == "correlation"]
    if corr_rows:
        st.subheader("Strong correlations (statistical)")
        st.dataframe(
            [
                {
                    "columns": " ↔ ".join(p.get("columns", [])[:2]),
                    "r": p.get("correlation"),
                    "direction": p.get("direction"),
                    "description": p.get("description"),
                }
                for p in corr_rows[:15]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("patterns.json"):
        st.json(json.loads(path.read_text()))


def _tab_anomalies(artifact_dir: Path) -> None:
    path = artifact_dir / "anomalies.json"
    if not path.exists():
        st.warning("No anomalies.json.")
        return

    result = load_anomalies(path)
    st.markdown(
        f"**{len(result.anomalies)}** flagged records · "
        f"explanation source: `{result.explanation_source}` · "
        f"features: {len(result.feature_columns)} numeric columns"
    )

    rows = []
    for a in result.anomalies:
        rows.append(
            {
                "entity": a.entity,
                "metric": a.metric,
                "severity": a.severity,
                "score": round(a.score, 3),
                "hypothesis": a.hypothesis,
                "action": a.recommended_action,
                "graph_id": a.graph_entity_id,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown(
        "**What you are seeing:** statistical outliers (Isolation Forest) — employees or rows "
        "whose numeric profile differs from peers. **Entity** is the employee ID (or row). "
        "**Metric** is the feature that deviates most."
    )

    critical = [a for a in result.anomalies if a.severity == "critical"]
    if critical:
        st.subheader("Critical — act now")
        for a in critical[:5]:
            entity_label = a.entity if a.entity.lower() != "unknown" else "Unlabeled row"
            st.markdown(
                f"""
<div style="background:#3f1d1d;border:1px solid #f87171;border-radius:8px;padding:12px;margin-bottom:8px;">
  <p style="color:#fecaca;margin:0;font-weight:600;">{entity_label} · {a.metric} (score {a.score:.2f})</p>
  <p style="color:#fef2f2;margin:8px 0 0;">{a.hypothesis}</p>
  <p style="color:#fca5a5;margin:8px 0 0;">→ {a.recommended_action}</p>
</div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("anomalies.json"):
        st.json(json.loads(path.read_text()))


def _display_image_file(path: Path, caption: str = "") -> None:
    """Show PNG/JPEG using bytes (more reliable than path strings in Streamlit)."""
    if not path.exists():
        return
    st.image(path.read_bytes(), caption=caption or None, use_container_width=True)


def _tab_forecast(artifact_dir: Path) -> None:
    path = artifact_dir / "forecast.json"
    if not path.exists():
        st.warning("No forecast.json — run the pipeline first.")
        return

    forecast = load_forecast(path)
    png = artifact_dir / "forecast.png"

    if forecast.status == "success":
        st.success("Time-series forecast completed")
        if forecast.forecast_narrative:
            st.markdown(forecast.forecast_narrative)
        if forecast.target_column:
            st.caption(f"Target: **{forecast.target_column}** · Time: **{forecast.time_column}**")
    elif forecast.status == "snapshot":
        st.success("Snapshot analytics — average metrics by group (HR / snapshot data)")
        if forecast.forecast_narrative:
            st.markdown(forecast.forecast_narrative)
        if forecast.target_column and forecast.time_column:
            st.caption(
                f"Comparing average **{forecast.target_column}** across **{forecast.time_column}**"
            )
    else:
        st.warning(forecast.user_message or "Forecast not available for this dataset.")
        if st.button("Generate forecast chart now", key="btn_refresh_forecast_tab"):
            with st.spinner("Building chart…"):
                msgs = refresh_visual_artifacts(artifact_dir, use_llm=False)
            for m in msgs.values():
                st.info(m)
            st.rerun()

    if png.exists():
        _display_image_file(png, "Forecast chart")
    elif forecast.status in ("success", "snapshot"):
        st.error("Chart file missing. Click **Refresh charts** in the sidebar or the button above.")

    if forecast.forecast:
        st.dataframe(
            [
                {
                    "group_or_date": p.date,
                    "value": p.value,
                    "lower": p.lower,
                    "upper": p.upper,
                }
                for p in forecast.forecast
            ],
            use_container_width=True,
            hide_index=True,
        )
    if forecast.prescriptive_actions:
        st.subheader("Recommended actions")
        for act in forecast.prescriptive_actions:
            st.markdown(f"- **{act.due_date}** — {act.action}")

    with st.expander("forecast.json"):
        st.json(json.loads(path.read_text()))


def _tab_graph(artifact_dir: Path) -> None:
    graph_json = artifact_dir / "graph.json"
    if not graph_json.exists():
        st.warning("No graph.json — run the pipeline first.")
        return

    try:
        graph_result = load_graph(graph_json)
    except Exception as exc:
        st.error(f"Could not load graph: {exc}")
        return

    node_count = graph_result.node_count
    st.markdown(
        f"**{node_count}** nodes · **{graph_result.edge_count}** edges · "
        f"source: `{graph_result.extraction_source}`"
    )

    if node_count == 0:
        st.error("Graph has no nodes.")
        if st.button("Build knowledge graph now", key="btn_refresh_graph_tab"):
            with st.spinner("Building graph…"):
                msgs = refresh_visual_artifacts(artifact_dir, use_llm=False)
            for m in msgs.values():
                st.info(m)
            st.rerun()
        return

    hubs = graph_result.hub_entities or []
    if hubs:
        st.caption(f"Hub entities: {', '.join(hubs[:5])}")

    st.markdown("#### Knowledge graph diagram")
    fig = render_graph_figure(graph_result.nodes, graph_result.edges)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    png = artifact_dir / "graph.png"
    if png.exists():
        with st.expander("Download-quality PNG", expanded=False):
            _display_image_file(png)

    html = artifact_dir / "graph.html"
    if html.exists() and node_count > 0:
        with st.expander("Interactive graph (optional)", expanded=False):
            components.html(html.read_text(encoding="utf-8"), height=520, scrolling=True)

    with st.expander("Node sample"):
        st.dataframe(
            [
                {
                    "id": n.id,
                    "label": n.label,
                    "type": n.node_type,
                    "centrality": n.centrality,
                    "anomaly": n.is_anomaly,
                }
                for n in graph_result.nodes[:25]
            ],
            use_container_width=True,
            hide_index=True,
        )


def _tab_report(artifact_dir: Path) -> None:
    html_report = artifact_dir / "report.html"
    pdf_path = artifact_dir / "executive_report.pdf"
    meta = load_report_meta(artifact_dir)

    if html_report.exists():
        st.markdown("**Executive HTML report** (embedded)")
        wrapped = _wrap_report_for_streamlit(html_report.read_text(encoding="utf-8"))
        components.html(
            f'<div style="background:#fff;padding:8px;">{wrapped}</div>',
            height=720,
            scrolling=True,
        )
    else:
        st.warning("No report.html.")

    st.divider()
    st.subheader("Downloads")
    d1, d2, d3, d4 = st.columns(4)

    if pdf_path.exists():
        d1.download_button(
            "PDF report",
            data=pdf_path.read_bytes(),
            file_name="executive_report.pdf",
            mime="application/pdf",
        )
    else:
        d1.info(
            "PDF needs WeasyPrint system libs (`brew install pango gdk-pixbuf libffi`). "
            "HTML report is always available."
        )

    if html_report.exists():
        d2.download_button(
            "HTML report",
            data=html_report.read_text(encoding="utf-8"),
            file_name="report.html",
            mime="text/html",
        )

    for label, fname in (
        ("patterns.json", "patterns.json"),
        ("anomalies.json", "anomalies.json"),
        ("graph.json", "graph.json"),
        ("cleaned.parquet", "cleaned.parquet"),
    ):
        p = artifact_dir / fname
        if p.exists():
            d3.download_button(
                label,
                data=p.read_bytes(),
                file_name=fname,
                mime="application/octet-stream",
                key=f"dl_{fname}",
            )

    if meta:
        with st.expander("report_meta.json"):
            st.json(meta)


def _show_results(artifact_dir: Path) -> None:
    refresh_key = f"visuals_refreshed_{artifact_dir.name}"
    if not st.session_state.get(refresh_key):
        with st.spinner("Rendering forecast & knowledge graph charts…"):
            msgs = refresh_visual_artifacts(artifact_dir, use_llm=False)
        st.session_state[refresh_key] = True
        for msg in msgs.values():
            if "failed" in msg.lower() or msg.startswith("No cleaned"):
                st.warning(msg)
            elif msg:
                st.toast(msg, icon="✅")

    _kpi_row(artifact_dir)
    st.divider()

    tabs = st.tabs(
        [
            "Overview",
            "0 · Ingest",
            "1 · Clean",
            "2 · Patterns",
            "3 · Anomalies",
            "4 · Forecast",
            "5 · Graph",
            "6 · Report",
        ]
    )
    with tabs[0]:
        _tab_overview(artifact_dir)
    with tabs[1]:
        _tab_ingest(artifact_dir)
    with tabs[2]:
        _tab_cleaning(artifact_dir)
    with tabs[3]:
        _tab_patterns(artifact_dir)
    with tabs[4]:
        _tab_anomalies(artifact_dir)
    with tabs[5]:
        _tab_forecast(artifact_dir)
    with tabs[6]:
        _tab_graph(artifact_dir)
    with tabs[7]:
        _tab_report(artifact_dir)


def main() -> None:
    st.set_page_config(page_title="Noise to Insight", page_icon="📊", layout="wide")

    st.title("AI Meets Data: From Noise to Insight")
    st.caption("Six-phase agentic pipeline — ingest through executive report")

    with st.sidebar:
        st.header("Configuration")
        if _azure_configured():
            st.success("Azure OpenAI configured")
        else:
            st.warning("Heuristic + statistical modes (no API key).")
        use_llm = st.toggle("Use LLM for all phases", value=_azure_configured())
        if st.session_state.get("artifact_dir"):
            if st.button("Refresh charts", help="Rebuild forecast & graph images from saved data"):
                ad = Path(st.session_state.artifact_dir)
                for key in list(st.session_state.keys()):
                    if key.startswith("visuals_refreshed_"):
                        del st.session_state[key]
                with st.spinner("Refreshing…"):
                    msgs = refresh_visual_artifacts(ad, use_llm=False)
                for m in msgs.values():
                    st.sidebar.success(m)
                st.rerun()
        st.divider()
        st.header("Data source")
        source = st.radio(
            "Input",
            ["Upload files", "Demo bundle", "Replay golden run"],
            index=0,
        )

    if "artifact_dir" not in st.session_state:
        st.session_state.artifact_dir = None
    if "phase_log" not in st.session_state:
        st.session_state.phase_log = []
    if "pipeline_error" not in st.session_state:
        st.session_state.pipeline_error = None

    input_paths: list[Path] | None = None

    if source == "Upload files":
        uploads = st.file_uploader(
            "CSV / JSON files",
            type=["csv", "json", "jsonl"],
            accept_multiple_files=True,
        )
        if uploads:
            input_paths, _ = _save_uploads(uploads)
            st.write("Files:", ", ".join(p.name for p in input_paths))
    elif source == "Demo bundle":
        input_paths = _demo_file_paths()
        st.info(f"Demo files: {', '.join(p.name for p in input_paths)}")
    else:
        st.info(f"Golden artifacts: `{GOLDEN_DIR}`")
        if not GOLDEN_DIR.exists():
            st.error("Golden run missing. Run pipeline once and copy output to data/artifacts/golden/.")

    run_clicked = st.button("Run full pipeline", type="primary", use_container_width=True)

    if run_clicked:
        st.session_state.pipeline_error = None
        st.session_state.phase_log = []
        try:
            if source == "Replay golden run":
                artifact_dir = replay_golden_run()
                st.session_state.artifact_dir = artifact_dir
                st.session_state.phase_log = [
                    {"phase": p, "status": "completed", "message": "golden replay"}
                    for p in PHASE_ORDER
                ]
                st.success(f"Golden run loaded → `{artifact_dir}`")
            else:
                if not input_paths:
                    st.error("No input files selected.")
                    st.stop()
                run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
                with st.status("Running six-phase pipeline…", expanded=True) as status:
                    final = run_pipeline(input_paths, run_id=run_id, use_llm=use_llm)
                    st.session_state.artifact_dir = final.artifact_dir
                    st.session_state.phase_log = [e.model_dump() for e in final.phase_log]
                    refresh_visual_artifacts(final.artifact_dir, use_llm=False)
                    for key in list(st.session_state.keys()):
                        if key.startswith("visuals_refreshed_"):
                            del st.session_state[key]
                    status.update(label="Pipeline complete", state="complete")
                st.success(f"Artifacts saved to `{final.artifact_dir}`")
        except Exception as exc:
            st.session_state.pipeline_error = str(exc)
            st.error(f"Pipeline failed: {exc}")

    if st.session_state.phase_log:
        st.divider()
        st.subheader("Pipeline progress")
        _render_stepper(st.session_state.phase_log)

    if st.session_state.artifact_dir:
        st.divider()
        _show_results(Path(st.session_state.artifact_dir))

    if st.session_state.pipeline_error:
        st.caption(f"Last error: {st.session_state.pipeline_error}")


if __name__ == "__main__":
    main()
