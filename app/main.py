"""Точка входа приложения.

Модуль связывает слои: создаёт общую инфраструктуру (хранилище, AI-клиент,
трейсер) на протяжении жизненного цикла, регистрирует роутеры и переводит
доменные исключения в HTTP-ответы.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.exceptions import (
    AIServiceError,
    AuthenticationError,
    AuthorizationError,
    EmailAlreadyRegisteredError,
    ResourceNotFoundError,
    VectorStoreError,
    WeakPasswordError,
)
from app.routers import router as api_router
from app.services import OpenAIClient, build_tracer
from app.vector import VectorStore

DESCRIPTION: str = (
    "Интерактивный терминал убежища из Left 4 Dead. Образы выживших отвечают на "
    "вопросы игрока, опираясь на архив лора в Qdrant и OpenAI API. Доступ к архиву "
    "открыт только учётным записям: токены доступа и обновления уезжают в cookie."
)

WWW_AUTHENTICATE: dict[str, str] = {"WWW-Authenticate": "Bearer"}
"""Заголовок отказа: он говорит клиенту, какую схему аутентификации здесь ждут."""


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Создать общие ресурсы при запуске и освободить их при остановке."""
    # The settings the application was built with win over the process-wide cache:
    # `create_app(settings)` must mean «this application runs on these settings»,
    # otherwise a test or a second instance would silently read someone else's .env.
    settings: Settings = getattr(application.state, "settings", None) or get_settings()
    if not settings.is_auth_configured:
        raise RuntimeError(
            "AUTH_JWT_SECRET не задан: без ключа подписи терминал не может выдавать "
            "токены. Сгенерировать ключ: "
            'python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )

    vector_store = VectorStore(settings)
    ai_client = OpenAIClient(settings)
    tracer = build_tracer(settings)

    if settings.auto_create_collections:
        await vector_store.ensure_collections()

    application.state.vector_store = vector_store
    application.state.ai_client = ai_client
    application.state.tracer = tracer

    try:
        yield
    finally:
        await ai_client.aclose()
        await vector_store.close()
        # The tracer batches observations, so the last few seconds of traces exist
        # only until the queue is drained; a shutdown that skips this loses them.
        tracer.flush()
        tracer.shutdown()


async def handle_resource_not_found(request: Request, exc: Exception) -> JSONResponse:
    """Перевести отсутствующую сущность в ``404 Not Found``."""
    assert isinstance(exc, ResourceNotFoundError)
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )


async def handle_ai_service_error(request: Request, exc: Exception) -> JSONResponse:
    """Перевести сбой провайдера в ``502 Bad Gateway``."""
    assert isinstance(exc, AIServiceError)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc)},
    )


async def handle_vector_store_error(request: Request, exc: Exception) -> JSONResponse:
    """Перевести сбой векторного хранилища в ``503 Service Unavailable``."""
    assert isinstance(exc, VectorStoreError)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


async def handle_authentication_error(request: Request, exc: Exception) -> JSONResponse:
    """Перевести неподтверждённый доступ в ``401 Unauthorized``.

    Вместе с текстом уезжает машинный код: по нему клиент отличает «обнови токен»
    от «иди на экран входа», не разбирая формулировку сообщения.
    """
    assert isinstance(exc, AuthenticationError)
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": str(exc), "code": exc.code},
        headers=WWW_AUTHENTICATE,
    )


async def handle_authorization_error(request: Request, exc: Exception) -> JSONResponse:
    """Перевести нехватку прав в ``403 Forbidden``."""
    assert isinstance(exc, AuthorizationError)
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"detail": str(exc), "code": exc.code},
    )


async def handle_email_taken(request: Request, exc: Exception) -> JSONResponse:
    """Перевести занятый адрес почты в ``409 Conflict``."""
    assert isinstance(exc, EmailAlreadyRegisteredError)
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": str(exc), "code": exc.code},
    )


async def handle_weak_password(request: Request, exc: Exception) -> JSONResponse:
    """Перевести пароль, не отвечающий политике, в ``422 Unprocessable Entity``."""
    assert isinstance(exc, WeakPasswordError)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": str(exc), "code": exc.code},
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Собрать приложение FastAPI; принимает настройки, чтобы упростить тесты."""
    active_settings: Settings = settings or get_settings()

    application = FastAPI(
        title=active_settings.app_name,
        version=active_settings.app_version,
        description=DESCRIPTION,
        debug=active_settings.debug,
        lifespan=lifespan,
    )
    application.state.settings = active_settings
    if active_settings.cors_allow_origins:
        # The frontend is served by its own dev server (or another host), so the
        # browser needs the API to advertise the origins it trusts. Credentials
        # are allowed because the session travels in cookies: without this flag the
        # browser would drop them on a cross-origin call. That also means the
        # wildcard origin is unusable — the list must stay explicit.
        application.add_middleware(
            CORSMiddleware,
            allow_origins=active_settings.cors_allow_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
    application.add_exception_handler(ResourceNotFoundError, handle_resource_not_found)
    application.add_exception_handler(AIServiceError, handle_ai_service_error)
    application.add_exception_handler(VectorStoreError, handle_vector_store_error)
    application.add_exception_handler(AuthenticationError, handle_authentication_error)
    application.add_exception_handler(AuthorizationError, handle_authorization_error)
    application.add_exception_handler(EmailAlreadyRegisteredError, handle_email_taken)
    application.add_exception_handler(WeakPasswordError, handle_weak_password)
    application.include_router(api_router, prefix=active_settings.api_prefix)

    @application.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        return {
            "service": active_settings.app_name,
            "version": active_settings.app_version,
            "docs": "/docs",
            "api_prefix": active_settings.api_prefix or "/",
        }
    return application


app = create_app()
