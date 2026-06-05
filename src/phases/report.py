"""Phase 6: Executive PDF report generation."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import ValidationError

from src.config import PROJECT_ROOT, get_settings
from src.llm.client import get_client, get_deployment_name
from src.models.anomalies import AnomalyDetectionResult
from src.models.forecast import ForecastResult
from src.models.graph import GraphBuildResult
from src.models.patterns import PatternDiscoveryResult
from src.models.report import ExecutiveReportContent
from src.phases.anomalies import load_anomalies
from src.phases.forecast import load_forecast
from src.phases.graph_build import load_graph
from src.phases.patterns import load_patterns

TEMPLATES_DIR = PROJECT_ROOT / "templates"
REPORT_TEMPLATE = "report.html"


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _image_to_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def gather_artifact_context(artifact_dir: Path) -> dict[str, Any]:
    """Load all phase artifacts from a run directory."""
    artifact_dir = artifact_dir.resolve()

    patterns: PatternDiscoveryResult | None = None
    anomalies: AnomalyDetectionResult | None = None
    forecast: ForecastResult | None = None
    graph: GraphBuildResult | None = None
    cleaning_report = _load_json_if_exists(artifact_dir / "cleaning_report.json")

    if (artifact_dir / "patterns.json").exists():
        patterns = load_patterns(artifact_dir / "patterns.json")
    if (artifact_dir / "anomalies.json").exists():
        anomalies = load_anomalies(artifact_dir / "anomalies.json")
    if (artifact_dir / "forecast.json").exists():
        forecast = load_forecast(artifact_dir / "forecast.json")
    if (artifact_dir / "graph.json").exists():
        graph = load_graph(artifact_dir / "graph.json")

    return {
        "artifact_dir": artifact_dir,
        "cleaning_report": cleaning_report,
        "patterns": patterns,
        "anomalies": anomalies,
        "forecast": forecast,
        "graph": graph,
        "forecast_chart": artifact_dir / "forecast.png",
    }


def build_kpis(ctx: dict[str, Any]) -> list[dict[str, str]]:
    """KPI strip for the report header."""
    patterns: PatternDiscoveryResult | None = ctx.get("patterns")
    anomalies: AnomalyDetectionResult | None = ctx.get("anomalies")
    forecast: ForecastResult | None = ctx.get("forecast")
    graph: GraphBuildResult | None = ctx.get("graph")
    cleaning = ctx.get("cleaning_report") or {}

    rows_after = cleaning.get("validation", {}).get("rows_after", "—")
    return [
        {"label": "Rows analyzed", "value": str(patterns.row_count if patterns else rows_after)},
        {
            "label": "Insights found",
            "value": str(len(patterns.insights) if patterns else 0),
        },
        {
            "label": "Anomalies flagged",
            "value": str(len(anomalies.anomalies) if anomalies else 0),
        },
        {
            "label": "Graph entities",
            "value": str(graph.node_count if graph else 0),
        },
        {
            "label": "Forecast",
            "value": (forecast.status if forecast else "n/a").upper(),
        },
    ]


def build_heuristic_executive_content(ctx: dict[str, Any]) -> ExecutiveReportContent:
    """Template executive summary when LLM is unavailable."""
    patterns: PatternDiscoveryResult | None = ctx.get("patterns")
    anomalies: AnomalyDetectionResult | None = ctx.get("anomalies")
    forecast: ForecastResult | None = ctx.get("forecast")
    graph: GraphBuildResult | None = ctx.get("graph")

    parts: list[str] = [
        "This report synthesizes cleaned operational data across pattern discovery, "
        "anomaly detection, forecasting, and knowledge graph analysis.",
    ]
    if patterns and patterns.insights:
        top = patterns.insights[0]
        parts.append(f"Primary insight: {top.title} — {top.summary}")
    if anomalies and anomalies.anomalies:
        top_a = anomalies.anomalies[0]
        parts.append(
            f"Highest-priority anomaly: {top_a.entity} on {top_a.metric} "
            f"({top_a.severity}): {top_a.hypothesis or 'requires review'}.",
        )
    if forecast and forecast.status in ("success", "snapshot") and forecast.forecast_narrative:
        parts.append(forecast.forecast_narrative)
    elif forecast and forecast.user_message:
        parts.append(forecast.user_message)
    if graph and graph.hub_entities:
        parts.append(f"Graph analysis centers on hubs: {', '.join(graph.hub_entities[:3])}.")

    actions: list[str] = []
    if anomalies:
        for a in anomalies.anomalies[:2]:
            if a.recommended_action:
                actions.append(a.recommended_action)
    if forecast and forecast.prescriptive_actions:
        for pa in forecast.prescriptive_actions[:2]:
            actions.append(f"{pa.action} (by {pa.due_date})")
    if not actions:
        actions = [
            "Validate data quality gates before the next planning cycle.",
            "Assign owners to top anomalies and track resolution weekly.",
        ]

    summary = " ".join(parts)[:1200]
    return ExecutiveReportContent(executive_summary=summary, top_actions=actions[:3])


def generate_executive_content_llm(ctx: dict[str, Any]) -> tuple[ExecutiveReportContent, str]:
    """LLM executive summary and action plan from phase artifacts."""
    settings = get_settings()
    if not settings.azure_openai_api_key:
        content = build_heuristic_executive_content(ctx)
        return content, "heuristic"

    compact: dict[str, Any] = {}
    patterns: PatternDiscoveryResult | None = ctx.get("patterns")
    anomalies: AnomalyDetectionResult | None = ctx.get("anomalies")
    forecast: ForecastResult | None = ctx.get("forecast")
    graph: GraphBuildResult | None = ctx.get("graph")

    if patterns:
        compact["insights"] = [i.model_dump() for i in patterns.insights[:5]]
    if anomalies:
        compact["anomalies"] = [a.model_dump() for a in anomalies.anomalies[:5]]
    if forecast:
        compact["forecast"] = {
            "status": forecast.status,
            "narrative": forecast.forecast_narrative,
            "actions": [a.model_dump() for a in forecast.prescriptive_actions],
            "forecast_tail": [p.model_dump() for p in forecast.forecast[-5:]],
        }
    if graph:
        compact["graph_hubs"] = graph.hub_entities
        compact["anomaly_nodes"] = [n.id for n in graph.nodes if n.is_anomaly][:5]

    system = (
        "Write a board-ready executive summary (max 250 words) and 3 top_actions bullets "
        "for operations and finance leaders. Be specific with entities, metrics, and dates "
        "from the evidence. Use decisive language."
    )

    client = get_client()
    deployment = get_deployment_name()

    try:
        completion = client.beta.chat.completions.parse(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(compact, default=str)},
            ],
            response_format=ExecutiveReportContent,
            temperature=0.2,
        )
        parsed = completion.choices[0].message.parsed
        if parsed:
            return parsed, "llm"
    except Exception:
        pass

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(compact, default=str)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        parsed = ExecutiveReportContent.model_validate_json(
            response.choices[0].message.content or "{}",
        )
        return parsed, "llm"
    except (ValidationError, Exception):
        return build_heuristic_executive_content(ctx), "heuristic"


def build_template_context(
    ctx: dict[str, Any],
    content: ExecutiveReportContent,
    *,
    team_name: str = "Noise to Insight",
    title: str = "Executive Intelligence Report",
) -> dict[str, Any]:
    """Assemble Jinja2 template variables."""
    patterns: PatternDiscoveryResult | None = ctx.get("patterns")
    anomalies: AnomalyDetectionResult | None = ctx.get("anomalies")
    forecast: ForecastResult | None = ctx.get("forecast")
    graph: GraphBuildResult | None = ctx.get("graph")
    forecast_chart: Path = ctx["forecast_chart"]

    insights = []
    if patterns:
        insights = [i.model_dump() for i in patterns.insights[:5]]

    anomaly_rows = []
    if anomalies:
        anomaly_rows = [a.model_dump() for a in anomalies.anomalies[:8]]

    forecast_section = None
    if forecast and forecast.status in ("success", "snapshot"):
        forecast_section = {
            "narrative": forecast.forecast_narrative or "",
            "actions": [a.model_dump() for a in forecast.prescriptive_actions],
        }
    elif forecast and forecast.user_message:
        forecast_section = {
            "narrative": forecast.user_message,
            "actions": [],
        }

    return {
        "title": title,
        "subtitle": "AI Meets Data: From Noise to Insight",
        "team_name": team_name,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "executive_summary": content.executive_summary,
        "top_actions": content.top_actions,
        "kpis": build_kpis(ctx),
        "insights": insights,
        "anomalies": anomaly_rows,
        "forecast_section": forecast_section,
        "forecast_chart_uri": _image_to_data_uri(forecast_chart),
        "graph_hubs": graph.hub_entities[:5] if graph else [],
    }


def render_report_html(
    template_ctx: dict[str, Any],
    output_path: Path,
) -> Path:
    """Render report HTML from Jinja template."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template(REPORT_TEMPLATE)
    html = template.render(**template_ctx)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def render_report_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Convert HTML to PDF via WeasyPrint. Returns False if system libs missing."""
    try:
        from weasyprint import HTML

        HTML(filename=str(html_path.resolve())).write_pdf(str(pdf_path.resolve()))
        return True
    except Exception:
        return False


def run_report(
    artifact_dir: Path,
    *,
    use_llm: bool = True,
    team_name: str = "Noise to Insight",
    title: str = "Executive Intelligence Report",
) -> Path:
    """
    Generate executive_report.pdf (and report.html) from phase artifacts.

    Expects artifact_dir to contain outputs from prior phases (patterns, anomalies,
    forecast, graph, optional cleaning_report, forecast.png).
    """
    artifact_dir = artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    ctx = gather_artifact_context(artifact_dir)

    if use_llm:
        content, source = generate_executive_content_llm(ctx)
    else:
        content = build_heuristic_executive_content(ctx)
        source = "heuristic"

    template_ctx = build_template_context(
        ctx,
        content,
        team_name=team_name,
        title=title,
    )
    template_ctx["narrative_source"] = source

    html_path = artifact_dir / "report.html"
    pdf_path = artifact_dir / "executive_report.pdf"

    render_report_html(template_ctx, html_path)
    pdf_ok = render_report_pdf(html_path, pdf_path)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "narrative_source": source,
        "html_path": html_path.name,
        "pdf_path": pdf_path.name if pdf_ok else None,
        "pdf_generated": pdf_ok,
        "executive_summary": content.executive_summary,
        "top_actions": content.top_actions,
    }
    (artifact_dir / "report_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    return pdf_path if pdf_ok else html_path


def load_report_meta(artifact_dir: Path) -> dict[str, Any]:
    """Load report_meta.json written by run_report."""
    path = artifact_dir / "report_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
