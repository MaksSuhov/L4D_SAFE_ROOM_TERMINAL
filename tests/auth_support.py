"""Общая оснастка тестов аутентификации: подмены зависимостей и вход через API."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import (
    get_clock,
    get_password_hasher,
    get_refresh_session_repository,
    get_user_repository,
)
from tests.fakes import (
    FakePasswordHasher,
    FixedClock,
    InMemoryRefreshSessionRepository,
    InMemoryUserRepository,
)

CSRF_HEADER: str = get_settings().auth_csrf_header_name
ACCESS_COOKIE: str = get_settings().auth_access_cookie_name
REFRESH_COOKIE: str = get_settings().auth_refresh_cookie_name
CSRF_COOKIE: str = get_settings().auth_csrf_cookie_name
REFRESH_COOKIE_PATH: str = get_settings().auth_refresh_cookie_path


def override_auth_dependencies(
    application: FastAPI,
    *,
    users: InMemoryUserRepository,
    sessions: InMemoryRefreshSessionRepository,
    hasher: FakePasswordHasher,
    clock: FixedClock,
) -> None:
    """Подменить двойниками всё, что сервис учётных записей берёт из окружения.

    Часы подменяются в том числе для выдачи токенов: сроки их жизни проверяются
    сдвигом часов, а не ожиданием в реальном времени.
    """
    application.dependency_overrides[get_user_repository] = lambda: users
    application.dependency_overrides[get_refresh_session_repository] = lambda: sessions
    application.dependency_overrides[get_password_hasher] = lambda: hasher
    application.dependency_overrides[get_clock] = lambda: clock


def log_in(
    client: TestClient,
    *,
    prefix: str,
    email: str,
    password: str,
) -> dict[str, Any]:
    """Войти через API и повторить CSRF-метку заголовком, как это делает браузер."""
    response = client.post(
        f"{prefix}/auth/token",
        data={"username": email, "password": password},
    )
    assert response.status_code == 200, response.text
    payload: dict[str, Any] = response.json()
    client.headers[CSRF_HEADER] = str(payload["csrf_token"])
    return payload


def register(
    client: TestClient,
    *,
    prefix: str,
    email: str,
    password: str,
    display_name: str | None = None,
) -> dict[str, Any]:
    """Зарегистрироваться через API и повторить CSRF-метку заголовком."""
    body: dict[str, Any] = {"email": email, "password": password}
    if display_name is not None:
        body["display_name"] = display_name
    response = client.post(f"{prefix}/auth/register", json=body)
    assert response.status_code == 201, response.text
    payload: dict[str, Any] = response.json()
    client.headers[CSRF_HEADER] = str(payload["csrf_token"])
    return payload
