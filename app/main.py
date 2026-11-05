"""Точка входа приложения.

Модуль связывает слои: создаёт общую инфраструктуру (движок, AI-клиент, трейсер)
на протяжении жизненного цикла, регистрирует роутеры и переводит доменные
исключения в HTTP-ответы.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.database import Database
from app.exceptions import AIServiceError, ResourceNotFoundError
from app.routers import router as api_router
from app.services import OpenAIClient, build_tracer

DESCRIPTION: str = (
    "Интерактивный терминал убежища из Left 4 Dead. Образы выживших отвечают на "
    "вопросы игрока, опираясь на архив лора в pgvector и OpenAI API."
)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Создать общие ресурсы при запуске и освободить их при остановке."""
    settings: Settings = get_settings()

    database = Database(settings)
    ai_client = OpenAIClient(settings)
    tracer = build_tracer(settings)

    if settings.auto_create_tables:
        await database.create_all()

    application.state.database = database
    application.state.ai_client = ai_client
    application.state.tracer = tracer

    try:
        yield
    finally:
        await ai_client.aclose()
        await database.dispose()
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
    if active_settings.cors_allow_origins:
        # The frontend is served by its own dev server (or another host), so the
        # browser needs the API to advertise the origins it trusts.
        application.add_middleware(
            CORSMiddleware,
            allow_origins=active_settings.cors_allow_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
    application.add_exception_handler(ResourceNotFoundError, handle_resource_not_found)
    application.add_exception_handler(AIServiceError, handle_ai_service_error)
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
