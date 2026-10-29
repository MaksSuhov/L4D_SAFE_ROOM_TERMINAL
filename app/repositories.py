"""Repository layer: the only place that knows SQL.

Repositories hide query construction behind intention-revealing methods, so the
service layer never touches ``select()`` or the session API directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Base
from app.interfaces import (
    CharacterRepositoryProtocol,
    LoreChunkRepositoryProtocol,
    RetrievedChunk,
)
from app.models import Character, LoreChunk

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """Persistence behaviour shared by every repository of a single model."""

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @property
    def session(self) -> AsyncSession:
        """The unit of work this repository participates in."""
        return self._session

    async def get_by_id(self, entity_id: int) -> ModelT | None:
        """Fetch an entity by primary key using the identity map when possible."""
        return await self._session.get(self.model, entity_id)

    async def add(self, instance: ModelT) -> ModelT:
        """Persist ``instance`` and refresh it so server defaults are populated."""
        self._session.add(instance)
        await self._session.flush()
        await self._session.refresh(instance)
        return instance

    async def count(self) -> int:
        """Return the number of rows of this model."""
        statement = select(func.count()).select_from(self.model)
        result = await self._session.execute(statement)
        return int(result.scalar_one())


class CharacterRepository(BaseRepository[Character], CharacterRepositoryProtocol):
    """Read access to the survivor personas stored in ``characters``."""

    model = Character

    async def get_all(self) -> Sequence[Character]:
        """Return every character ordered by identifier."""
        statement = select(Character).order_by(Character.id)
        result = await self._session.execute(statement)
        return result.scalars().all()

    async def get_by_name(self, name: str) -> Character | None:
        """Look up a character by its unique display name."""
        statement = select(Character).where(Character.name == name)
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()


class LoreChunkRepository(BaseRepository[LoreChunk], LoreChunkRepositoryProtocol):
    """Storage and cosine-similarity search over the lore knowledge base."""

    model = LoreChunk

    async def create_chunk(
        self,
        *,
        content: str,
        data_type: str,
        embedding: Sequence[float],
    ) -> LoreChunk:
        """Persist one embedded lore fragment."""
        chunk = LoreChunk(content=content, data_type=data_type, embedding=list(embedding))
        return await self.add(chunk)

    async def create_chunks(
        self,
        *,
        contents: Sequence[str],
        data_types: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> Sequence[LoreChunk]:
        """Persist a batch of embedded lore fragments in a single flush."""
        if not (len(contents) == len(data_types) == len(embeddings)):
            raise ValueError("contents, data_types and embeddings must have the same length")

        chunks = [
            LoreChunk(content=content, data_type=data_type, embedding=list(embedding))
            for content, data_type, embedding in zip(contents, data_types, embeddings, strict=True)
        ]
        self._session.add_all(chunks)
        await self._session.flush()
        return chunks

    async def search_similar_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        data_type: str | None = None,
    ) -> Sequence[RetrievedChunk]:
        """Return the ``limit`` nearest chunks ordered by descending cosine similarity.

        pgvector exposes the cosine *distance* (``1 - similarity``); the operator
        ``<=>`` is rendered by ``cosine_distance`` and can be accelerated by the
        HNSW index declared on the model.
        """
        distance = LoreChunk.embedding.cosine_distance(list(query_embedding)).label(
            "cosine_distance"
        )
        statement = select(LoreChunk, distance).order_by(distance).limit(limit)
        if data_type is not None:
            statement = statement.where(LoreChunk.data_type == data_type)

        result = await self._session.execute(statement)
        return [
            RetrievedChunk(chunk=chunk, similarity=1.0 - float(distance_value))
            for chunk, distance_value in result.all()
        ]
