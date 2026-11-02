"""HTTP-слой: только маршрутизация, валидация запросов и перевод ошибок."""

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
    summary="Проверка готовности",
)
async def health(settings: SettingsDep, session: SessionDep) -> HealthResponse:
    """Сообщить о живости процесса вместе с доступностью PostgreSQL."""
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
    summary="Список выживших, доступных на терминале",
)
async def list_characters(characters: CharacterRepositoryDep) -> Sequence[Character]:
    """Вернуть каждый образ, в который терминал умеет перевоплощаться."""
    return await characters.get_all()


@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Задать вопрос выжившему",
    responses={404: {"description": "Запрошенного персонажа нет"}},
)
async def chat(payload: ChatRequest, rag_service: RAGServiceDep) -> ChatResponse:
    """Прогнать RAG-конвейер и ответить от имени запрошенного персонажа."""
    return await rag_service.process_chat_message(payload.character_id, payload.message)
