"""Azure OpenAI client factory."""

from openai import AzureOpenAI

from src.config import get_settings


def get_client() -> AzureOpenAI:
    """Return a configured Azure OpenAI client."""
    settings = get_settings()
    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )


def get_deployment_name() -> str:
    """Return the configured model deployment name."""
    return get_settings().azure_openai_deployment
