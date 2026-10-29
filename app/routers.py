"""HTTP layer: routing, request validation and error translation only."""

from __future__ import annotations

from collections.abc import Sequence

from fastapi import APIRouter
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.dependencies import (
    CharacterRepositoryDep,
    RAGServiceDep,
    SessionDep,
    SettingsDep,
)
from app.models import Character
from app.schemas import CharacterRead, ChatRequest, ChatResponse, HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Readiness probe",
)
async def health(settings: SettingsDep, session: SessionDep) -> HealthResponse:
    """Report process liveness together with the reachability of PostgreSQL."""
    try:
        await session.execute(text("SELECT 1"))
        database_status = "ok"
    except (SQLAlchemyError, OSError):
        # An unreachable server can surface either as a driver error or as a raw
        # socket error, and both mean the same thing for the probe.
        await session.rollback()
        database_status = "unavailable"

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        database=database_status,
    )


@router.get(
    "/characters",
    response_model=list[CharacterRead],
    tags=["characters"],
    summary="List the survivors available at the terminal",
)
async def list_characters(characters: CharacterRepositoryDep) -> Sequence[Character]:
    """Return every persona the terminal can impersonate."""
    return await characters.get_all()


@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Ask a survivor a question",
    responses={404: {"description": "The requested character does not exist"}},
)
async def chat(payload: ChatRequest, rag_service: RAGServiceDep) -> ChatResponse:
    """Run the RAG pipeline and answer as the requested character."""
    return await rag_service.process_chat_message(payload.character_id, payload.message)
