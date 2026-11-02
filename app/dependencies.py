"""Провайдеры внедрения зависимостей для слоя API.

Здесь строится каждый объект, нужный роутерам, и внедряется через ``Depends``:
сессия живёт на протяжении запроса, репозитории привязаны к этой сессии, а
RAG-сервис получает те абстракции, которые объявляет.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.database import Database
from app.exceptions import AIServiceError
from app.interfaces import AIClientProtocol, QueryExpanderProtocol
from app.repositories import CharacterRepository, LoreChunkRepository
from app.services import (
    ArchivePromptBuilder,
    HideQueryExpander,
    NoQueryExpander,
    RAGService,
    RetrievalPolicy,
    TierPolicy,
)


async def get_database(request: Request) -> Database:
    """Вернуть дескриптор базы, созданный жизненным циклом приложения."""
    database: Database | None = getattr(request.app.state, "database", None)
    if database is None:
        raise RuntimeError("Database is not initialised; the application lifespan did not run")
    return database


async def get_session(
    database: Annotated[Database, Depends(get_database)],
) -> AsyncIterator[AsyncSession]:
    """Отдать сессию в пределах запроса и закрыть её после формирования ответа."""
    async with database.session() as session:
        yield session


async def get_ai_client(request: Request) -> AIClientProtocol:
    """Вернуть общий AI-клиент, созданный жизненным циклом приложения."""
    client: AIClientProtocol | None = getattr(request.app.state, "ai_client", None)
    if client is None:
        raise AIServiceError("AI client is not initialised; the application lifespan did not run")
    return client


def get_character_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> CharacterRepository:
    """Привязать репозиторий персонажей к сессии в пределах запроса."""
    return CharacterRepository(session)


def get_lore_chunk_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> LoreChunkRepository:
    """Привязать репозиторий лора к сессии в пределах запроса."""
    return LoreChunkRepository(session)


def get_prompt_builder() -> ArchivePromptBuilder:
    """Вернуть сборщик многоуровневого брифинга по архиву.

    Он не хранит состояния, так что одного экземпляра хватило бы на весь процесс;
    провайдер сохраняет единообразие графа зависимостей и позволяет тривиально
    подменить политику промптов из теста.
    """
    return ArchivePromptBuilder()


def build_query_expander(ai_client: AIClientProtocol, settings: Settings) -> QueryExpanderProtocol:
    """Собрать расширитель запроса по настройкам: HIDE или его выключенную заглушку.

    Отдельная функция нужна офлайн-стенду оценки: он собирает те же зависимости, что
    и HTTP-слой, но без графа ``Depends``.
    """
    if not settings.hide_enabled:
        return NoQueryExpander()
    return HideQueryExpander(ai_client, max_questions=settings.hide_max_questions)


def get_query_expander(
    ai_client: Annotated[AIClientProtocol, Depends(get_ai_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> QueryExpanderProtocol:
    """Вернуть расширитель запроса, общий для всех ходов разговора."""
    return build_query_expander(ai_client, settings)


def get_rag_service(
    character_repository: Annotated[CharacterRepository, Depends(get_character_repository)],
    lore_chunk_repository: Annotated[LoreChunkRepository, Depends(get_lore_chunk_repository)],
    prompt_builder: Annotated[ArchivePromptBuilder, Depends(get_prompt_builder)],
    ai_client: Annotated[AIClientProtocol, Depends(get_ai_client)],
    settings: Annotated[Settings, Depends(get_settings)],
    query_expander: Annotated[QueryExpanderProtocol, Depends(get_query_expander)],
) -> RAGService:
    """Собрать RAG-конвейер из его зависимостей и конфигурации."""
    return RAGService(
        character_repository,
        lore_chunk_repository,
        prompt_builder,
        ai_client,
        policy=RetrievalPolicy(
            official=TierPolicy(
                limit=settings.rag_official_top_k,
                min_similarity=settings.rag_min_similarity,
            ),
            graffiti=TierPolicy(
                limit=settings.rag_graffiti_top_k,
                min_similarity=settings.rag_min_similarity,
            ),
            voice=TierPolicy(
                limit=settings.rag_voice_top_k,
                min_similarity=settings.rag_voice_min_similarity,
            ),
        ),
        query_expander=query_expander,
    )


SessionDep = Annotated[AsyncSession, Depends(get_session)]
CharacterRepositoryDep = Annotated[CharacterRepository, Depends(get_character_repository)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
