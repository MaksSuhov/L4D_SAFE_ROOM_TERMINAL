from __future__ import annotations

from collections.abc import Sequence

from fastapi import APIRouter, Depends

from app.dependencies import (
    CharacterRepositoryDep,
    CurrentUserDep,
    RAGServiceDep,
    SettingsDep,
    VectorStoreDep,
    require_csrf_token,
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
    """Сообщить о живости процесса вместе с доступностью векторного хранилища.

    Проба остаётся открытой: по ней убеждаются, что терминал вообще поднялся, —
    и проверяющий, у которого нет учётной записи, тоже.
    """
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
    responses={401: {"description": "Запрос не предъявил действующую сессию"}},
)
async def list_characters(
    characters: CharacterRepositoryDep,
    user: CurrentUserDep,
) -> Sequence[Character]:
    """Вернуть каждый образ, в который терминал умеет перевоплощаться."""
    return await characters.get_all()


@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Задать вопрос выжившему",
    responses={
        401: {"description": "Запрос не предъявил действующую сессию"},
        403: {"description": "Изменяющий запрос без CSRF-метки"},
        404: {"description": "Запрошенного персонажа нет"},
    },
    dependencies=[Depends(require_csrf_token)],
)
async def chat(
    payload: ChatRequest,
    rag_service: RAGServiceDep,
    user: CurrentUserDep,
) -> ChatResponse:
    """Прогнать RAG-конвейер и ответить от имени запрошенного персонажа.

    Вопрос требует учётной записи: за одним ответом стоит вызов провайдера, а он
    оплачивается владельцем терминала. Персонаж при этом отвечает одинаково всем —
    разговоры не сохраняются и к учётной записи не привязаны.
    """
    return await rag_service.process_chat_message(payload.character_id, payload.message)
