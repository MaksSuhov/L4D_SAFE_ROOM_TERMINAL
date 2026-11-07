"""Конфигурация тестов.

Переменные окружения выставляются до импорта ``app``, чтобы кэш настроек и
создание клиента OpenAI видели детерминированную конфигурацию.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("AI_API_KEY", "test-key")
os.environ.setdefault("AI_BASE_URL", "http://localhost:1/v1")
os.environ.setdefault("RAG_MIN_SIMILARITY", "0.0")
# A developer's .env carries real Langfuse keys, and the settings cache would pick
# them up, so a test run would ship traces to an external project. The switch is
# forced off here, before the first Settings is built.
os.environ.setdefault("LANGFUSE_ENABLED", "false")
# Qdrant's version check turns client construction into a network call. Tests build
# the graph too, and they must stay hermetic, so the check is forced off here.
os.environ.setdefault("QDRANT_CHECK_COMPATIBILITY", "false")
# The signing secret has no default: the application refuses to start without it.
# Tests bring their own, so the run neither depends on a developer's .env nor
# writes to it.
os.environ.setdefault("AUTH_JWT_SECRET", "test-secret-that-is-long-enough-to-sign")

from app.config import get_settings  # noqa: E402
from tests.fakes import (  # noqa: E402
    FakePasswordHasher,
    FixedClock,
    InMemoryRefreshSessionRepository,
    InMemoryUserRepository,
)


@pytest.fixture(scope="session", autouse=True)
def settings_cache() -> Iterator[None]:
    """Пересобрать настройки из тестового окружения для этой сессии."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def users() -> InMemoryUserRepository:
    """Пустое хранилище учётных записей: тесты наполняют его тем, что им нужно."""
    return InMemoryUserRepository()


@pytest.fixture
def sessions() -> InMemoryRefreshSessionRepository:
    """Пустое хранилище refresh-сессий."""
    return InMemoryRefreshSessionRepository()


@pytest.fixture
def hasher() -> FakePasswordHasher:
    """Хешер-двойник: Argon2 проверяется отдельно и не тормозит остальные тесты."""
    return FakePasswordHasher()


@pytest.fixture
def clock() -> FixedClock:
    """Часы-двойник: сроки жизни токенов проверяются сдвигом, а не ожиданием."""
    return FixedClock()
