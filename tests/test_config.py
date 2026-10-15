"""Unit tests for configuration loading and environment overrides."""

import pytest
from app.config import Settings, settings


def test_default_config(monkeypatch):
    """Test that default configuration values match specifications."""
    # Ensure no relevant env vars are set
    for env_var in [
        "CHAT_BASE_URL",
        "CHAT_MODEL",
        "CHAT_API_KEY",
        "CHAT_TOKEN_BUDGET",
        "EMBED_BASE_URL",
        "EMBED_MODEL",
        "EMBED_API_KEY",
        "RERANK_BASE_URL",
        "RERANK_MODEL",
        "RERANK_API_KEY",
    ]:
        monkeypatch.delenv(env_var, raising=False)

    conf = Settings(_env_file=None)
    assert conf.chat_base_url == "http://8.219.192.36:9000"
    assert conf.chat_model == "claude-opus-4-8"
    assert conf.chat_api_key == "test-token"
    assert conf.chat_token_budget == 2000

    assert conf.embed_base_url is None
    assert conf.embed_model is None
    assert conf.embed_api_key is None

    assert conf.rerank_base_url is None
    assert conf.rerank_model is None
    assert conf.rerank_api_key is None


def test_env_override(monkeypatch):
    """Test that environment variables correctly override default configurations."""
    monkeypatch.setenv("CHAT_BASE_URL", "http://custom-host:9999")
    monkeypatch.setenv("CHAT_MODEL", "custom-model")
    monkeypatch.setenv("CHAT_API_KEY", "custom-token-123")
    monkeypatch.setenv("CHAT_TOKEN_BUDGET", "4000")
    monkeypatch.setenv("EMBED_BASE_URL", "http://embed-host:9001")
    monkeypatch.setenv("EMBED_MODEL", "bge-m3")
    monkeypatch.setenv("EMBED_API_KEY", "embed-key-456")
    monkeypatch.setenv("RERANK_BASE_URL", "http://rerank-host:9002")
    monkeypatch.setenv("RERANK_MODEL", "bge-reranker-large")
    monkeypatch.setenv("RERANK_API_KEY", "rerank-key-789")

    conf = Settings(_env_file=None)
    assert conf.chat_base_url == "http://custom-host:9999"
    assert conf.chat_model == "custom-model"
    assert conf.chat_api_key == "custom-token-123"
    assert conf.chat_token_budget == 4000
    assert isinstance(conf.chat_token_budget, int)

    assert conf.embed_base_url == "http://embed-host:9001"
    assert conf.embed_model == "bge-m3"
    assert conf.embed_api_key == "embed-key-456"

    assert conf.rerank_base_url == "http://rerank-host:9002"
    assert conf.rerank_model == "bge-reranker-large"
    assert conf.rerank_api_key == "rerank-key-789"


def test_extra_env_vars_ignored(monkeypatch):
    """Test that extra unconfigured env vars are ignored without error."""
    monkeypatch.setenv("UNRECOGNIZED_CUSTOM_ENV_VAR", "value")
    conf = Settings(_env_file=None)
    assert not hasattr(conf, "unrecognized_custom_env_var")


def test_settings_singleton_instance():
    """Test that exported settings is an instance of Settings."""
    assert isinstance(settings, Settings)
