"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEMO_DIR = DATA_DIR / "demo"
ARTIFACTS_DIR = DATA_DIR / "artifacts"


@dataclass(frozen=True)
class Settings:
    """Azure OpenAI and path settings."""

    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment: str
    azure_openai_api_version: str


def get_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings(
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    )


def ensure_artifact_dir(run_id: str) -> Path:
    """Create and return the artifact directory for a pipeline run."""
    path = ARTIFACTS_DIR / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path
