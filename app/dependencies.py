"""Dependency-injection providers for the API layer.

Every object the routers need is built here and injected with ``Depends``: the
session lives for the duration of a request, repositories are bound to that
session, and the RAG service receives the abstractions it declares.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.database import Database
from app.exceptions import AIServiceError
from app.interfaces import AIClientProtocol
from app.repositories import CharacterRepository, LoreChunkRepository
from app.services import RAGService


async def get_database(request: Request) -> Database:
    """Return the database handle created by the application lifespan."""
    database: Database | None = getattr(request.app.state, "database", None)
    if database is None:
        raise RuntimeError("Database is not initialised; the application lifespan did not run")
    return database


async def get_session(
    database: Annotated[Database, Depends(get_database)],
) -> AsyncIterator[AsyncSession]:
    """Yield a request-scoped session and close it once the response is produced."""
    async with database.session() as session:
        yield session


async def get_ai_client(request: Request) -> AIClientProtocol:
    """Return the shared AI client created by the application lifespan."""
    client: AIClientProtocol | None = getattr(request.app.state, "ai_client", None)
    if client is None:
        raise AIServiceError("AI client is not initialised; the application lifespan did not run")
    return client


def get_character_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> CharacterRepository:
    """Bind a character repository to the request-scoped session."""
    return CharacterRepository(session)


def get_lore_chunk_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> LoreChunkRepository:
    """Bind a lore repository to the request-scoped session."""
    return LoreChunkRepository(session)


def get_rag_service(
    character_repository: Annotated[CharacterRepository, Depends(get_character_repository)],
    lore_chunk_repository: Annotated[LoreChunkRepository, Depends(get_lore_chunk_repository)],
    ai_client: Annotated[AIClientProtocol, Depends(get_ai_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RAGService:
    """Assemble the RAG pipeline from its collaborators and the configuration."""
    return RAGService(
        character_repository,
        lore_chunk_repository,
        ai_client,
        top_k=settings.rag_top_k,
        min_similarity=settings.rag_min_similarity,
    )


SessionDep = Annotated[AsyncSession, Depends(get_session)]
CharacterRepositoryDep = Annotated[CharacterRepository, Depends(get_character_repository)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
