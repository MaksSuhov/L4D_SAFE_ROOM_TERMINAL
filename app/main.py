"""Application entry point.

The module wires the layers together: it creates the shared infrastructure
(engine, AI client) during the lifespan, registers the routers and translates
domain exceptions into HTTP responses.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.database import Database
from app.exceptions import AIServiceError, ResourceNotFoundError
from app.routers import router as api_router
from app.services import OpenAIClient

DESCRIPTION: str = (
    "Interactive safe-room terminal from Left 4 Dead. Survivor personas answer "
    "player questions using a pgvector-backed lore archive and the OpenAI API."
)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Create shared resources on start-up and release them on shutdown."""
    settings: Settings = get_settings()

    database = Database(settings)
    ai_client = OpenAIClient(settings)

    if settings.auto_create_tables:
        await database.create_all()

    application.state.database = database
    application.state.ai_client = ai_client

    try:
        yield
    finally:
        await ai_client.aclose()
        await database.dispose()


async def handle_resource_not_found(request: Request, exc: Exception) -> JSONResponse:
    """Translate a missing entity into ``404 Not Found``."""
    assert isinstance(exc, ResourceNotFoundError)
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )


async def handle_ai_service_error(request: Request, exc: Exception) -> JSONResponse:
    """Translate a provider failure into ``502 Bad Gateway``."""
    assert isinstance(exc, AIServiceError)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc)},
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application; accepts settings to ease testing."""
    active_settings: Settings = settings or get_settings()

    application = FastAPI(
        title=active_settings.app_name,
        version=active_settings.app_version,
        description=DESCRIPTION,
        debug=active_settings.debug,
        lifespan=lifespan,
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
