"""Smoke tests: verify package structure and imports."""

import importlib
from importlib.metadata import PackageNotFoundError, version


def test_import_root_package() -> None:
    import src

    assert src.__version__ == "0.1.0"


def test_import_config() -> None:
    from src.config import ARTIFACTS_DIR, DEMO_DIR, PROJECT_ROOT, get_settings

    assert PROJECT_ROOT.name in ("Build", "noise-to-insight")
    assert DEMO_DIR.parent.name == "data"
    settings = get_settings()
    assert hasattr(settings, "azure_openai_deployment")


def test_import_llm_client() -> None:
    from src.llm.client import get_client, get_deployment_name

    assert callable(get_client)
    assert callable(get_deployment_name)


def test_import_ingest_stubs() -> None:
    from src.ingest import loader, profiler

    assert hasattr(loader, "load_files")
    assert hasattr(profiler, "profile_dataset")


def test_import_phase_stubs() -> None:
    modules = [
        "src.phases.cleaning",
        "src.phases.patterns",
        "src.phases.anomalies",
        "src.phases.forecast",
        "src.phases.graph_build",
        "src.phases.report",
    ]
    for name in modules:
        mod = importlib.import_module(name)
        assert mod is not None


def test_import_orchestrator_and_viz() -> None:
    from src.orchestrator import graph as orchestrator_graph
    from src.viz import graph_viz

    assert hasattr(orchestrator_graph, "run_pipeline")
    assert hasattr(graph_viz, "render_graph_html")


def test_import_models() -> None:
    from src.models import PipelineState
    from src.models.state import AnomalyRecord, GraphPayload, InsightCard

    assert PipelineState is not None
    assert InsightCard.model_fields["title"] is not None
    assert AnomalyRecord.model_fields["score"] is not None
    assert GraphPayload.model_fields["nodes"] is not None


def test_third_party_dependencies_importable() -> None:
    """Ensure declared runtime dependencies resolve after install."""
    for module in (
        "polars",
        "duckdb",
        "pydantic",
        "langgraph",
        "langchain_openai",
        "openai",
        "dotenv",
        "streamlit",
        "networkx",
        "pyvis",
        "plotly",
        "matplotlib",
        "sklearn",
        "prophet",
        "jinja2",
    ):
        importlib.import_module(module)

    # WeasyPrint is installed as a wheel but needs system libs (Pango) to import.
    try:
        version("weasyprint")
    except PackageNotFoundError as exc:
        raise AssertionError("weasyprint package not installed") from exc
