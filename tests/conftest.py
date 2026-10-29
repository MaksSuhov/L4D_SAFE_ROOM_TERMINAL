"""Test configuration.

Environment variables are set before ``app`` is imported so that the settings
cache and the OpenAI client construction see a deterministic configuration.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:1/v1")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://l4d:l4d@localhost:5432/l4d_terminal",
)
os.environ.setdefault("RAG_MIN_SIMILARITY", "0.0")

from app.config import get_settings  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def settings_cache() -> Iterator[None]:
    """Rebuild settings from the test environment for this session."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
