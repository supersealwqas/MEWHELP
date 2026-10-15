"""Pytest shared fixtures and test configuration."""

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure tests run with predictable environment variables."""
    # Specific tests can override using monkeypatch as needed
    yield
