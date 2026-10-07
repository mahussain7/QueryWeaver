"""Groq selection must coexist with credentials for a separate embedding provider."""

from pathlib import Path
import runpy

import pytest

from api.core.pipeline import validate_custom_model


@pytest.fixture
def load_config(monkeypatch):
    """Evaluate an isolated config without reading developer credentials or rebinding imports."""
    for name in (
        "OLLAMA_MODEL", "GROQ_API_KEY", "GROQ_MODEL", "OPENAI_API_KEY", "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY", "COHERE_API_KEY", "AZURE_API_KEY", "VOYAGE_API_KEY",
        "COMPLETION_MODEL", "EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)

    def load(**env):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return runpy.run_path(str(Path(__file__).parents[1] / "api" / "config.py"))["Config"]

    return load


def test_groq_requires_embeddings(load_config):
    with pytest.raises(ValueError, match="Set EMBEDDING_MODEL.*VOYAGE_API_KEY"):
        load_config(GROQ_API_KEY="test-groq-key")


def test_groq_defaults_with_voyage(load_config):
    config = load_config(GROQ_API_KEY="test-groq-key", VOYAGE_API_KEY="test-voyage-key")
    assert config.LLM_PROVIDER == "groq"
    assert config.AZURE_FLAG is False
    assert config.COMPLETION_MODEL == "groq/openai/gpt-oss-120b"
    assert config.EMBEDDING_MODEL.model_name == "voyage/voyage-3"


@pytest.mark.parametrize("embedding_key,embedding_model", [
    ("OPENAI_API_KEY", "openai/text-embedding-ada-002"),
    ("GEMINI_API_KEY", "gemini/gemini-embedding-001"),
    ("COHERE_API_KEY", "cohere/embed-v4.0"),
    ("AZURE_API_KEY", "azure/text-embedding-ada-002"),
])
def test_embedding_credentials_do_not_override_groq(load_config, embedding_key, embedding_model):
    config = load_config(
        GROQ_API_KEY="test-groq-key", EMBEDDING_MODEL=embedding_model,
        VOYAGE_API_KEY="test-voyage-key", **{embedding_key: "test-embedding-key"},
    )
    assert config.LLM_PROVIDER == "groq"
    assert config.COMPLETION_MODEL == "groq/openai/gpt-oss-120b"
    assert config.EMBEDDING_MODEL.model_name == embedding_model


@pytest.mark.parametrize("model", ["openai/gpt-oss-20b", "groq/openai/gpt-oss-20b"])
def test_groq_model_accepts_optional_provider_prefix(load_config, model):
    config = load_config(
        GROQ_API_KEY="test-groq-key", GROQ_MODEL=model, EMBEDDING_MODEL="ollama/nomic-embed-text",
    )
    assert config.COMPLETION_MODEL == "groq/openai/gpt-oss-20b"


def test_completion_override_wins_over_groq_model(load_config):
    config = load_config(
        GROQ_API_KEY="test-groq-key", GROQ_MODEL="openai/gpt-oss-120b",
        COMPLETION_MODEL="groq/openai/gpt-oss-20b", EMBEDDING_MODEL="ollama/nomic-embed-text",
    )
    assert config.COMPLETION_MODEL == "groq/openai/gpt-oss-20b"


def test_ollama_keeps_priority(load_config):
    config = load_config(OLLAMA_MODEL="llama3.1", GROQ_API_KEY="test-groq-key")
    assert config.LLM_PROVIDER == "ollama"
    assert config.COMPLETION_MODEL == "ollama/llama3.1"


def test_existing_cloud_selection_without_groq(load_config):
    config = load_config(OPENAI_API_KEY="test-openai-key", GEMINI_API_KEY="test-gemini-key")
    assert config.LLM_PROVIDER == "openai"


def test_groq_custom_model_passes_pipeline_validation():
    validate_custom_model("groq/openai/gpt-oss-120b")
