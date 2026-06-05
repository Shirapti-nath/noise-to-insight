"""Pipeline phases: cleaning through executive report."""

from src.phases.anomalies import get_anomalies_for_entity, link_anomaly_to_graph_entity, run_anomalies
from src.phases.forecast import load_forecast, run_forecast
from src.phases.graph_build import highlight_path, load_graph, run_graph_build
from src.phases.patterns import get_top_insights, run_patterns
from src.phases.report import load_report_meta, run_report

__all__ = [
    "get_anomalies_for_entity",
    "get_top_insights",
    "highlight_path",
    "link_anomaly_to_graph_entity",
    "load_forecast",
    "load_graph",
    "load_report_meta",
    "run_anomalies",
    "run_forecast",
    "run_graph_build",
    "run_patterns",
    "run_report",
]
