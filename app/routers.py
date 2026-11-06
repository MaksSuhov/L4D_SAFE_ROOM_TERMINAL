"""HTTP-слой: только маршрутизация, валидация запросов и перевод ошибок."""

from __future__ import annotations

from collections.abc import Sequence

from fastapi import APIRouter

from app.dependencies import (
    CharacterRepositoryDep,
    RAGServiceDep,
    SettingsDep,
    VectorStoreDep,
)
from app.domain import Character
from app.schemas import CharacterRead, ChatRequest, ChatResponse, HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Проверка готовности",
)
async def health(settings: SettingsDep, vector_store: VectorStoreDep) -> HealthResponse:
    """Сообщить о живости процесса вместе с доступностью векторного хранилища."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        vector_store="ok" if await vector_store.ping() else "unavailable",
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
