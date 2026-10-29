"""Repository layer: the only place that knows SQL.

Repositories hide query construction behind intention-revealing methods, so the
service layer never touches ``select()`` or the session API directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, TypeVar

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Base
from app.interfaces import (
    CharacterRepositoryProtocol,
    LoreChunkRepositoryProtocol,
    RetrievedChunk,
)
from app.models import Character, LoreChunk
from app.schemas import EmbeddedLoreChunk
from app.taxonomy import SourceType

ModelT = TypeVar("ModelT", bound=Base)

HASH_LOOKUP_CHUNK_SIZE = 1000
"""Upper bound on the parameters of one ``IN (...)`` lookup, so a large corpus
never produces a statement the server refuses to plan."""


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

    async def create_chunk(self, record: EmbeddedLoreChunk) -> LoreChunk:
        """Persist one embedded lore fragment."""
        return await self.add(self._to_model(record))

    async def create_chunks(self, records: Sequence[EmbeddedLoreChunk]) -> Sequence[LoreChunk]:
        """Persist a batch of embedded lore fragments in a single flush."""
        if not records:
            return []
        chunks = [self._to_model(record) for record in records]
        self._session.add_all(chunks)
        await self._session.flush()
        return chunks

    async def search_similar_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        source_types: Sequence[SourceType] | None = None,
        topic: str | None = None,
        character_name: str | None = None,
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
        if source_types is not None:
            statement = statement.where(
                LoreChunk.source_type.in_([str(source_type) for source_type in source_types])
            )
        if topic is not None:
            statement = statement.where(LoreChunk.topic == topic)
        if character_name is not None:
            statement = statement.where(LoreChunk.character_name == character_name)

        result = await self._session.execute(statement)
        return [
            RetrievedChunk(chunk=chunk, similarity=1.0 - float(distance_value))
            for chunk, distance_value in result.all()
        ]

    async def delete_all(self) -> int:
        """Remove every stored fragment and report how many rows went.

        Ingestion is incremental — a record whose text changed is stored again
        under a new hash rather than replacing the old row — so re-indexing a
        corpus that was edited in place needs an explicit wipe first. The method
        is deliberately absent from the repository port: no service is allowed to
        empty the archive, only the seeding tool that owns its lifecycle.
        """
        result = await self._session.execute(delete(LoreChunk))
        # A DELETE returns a cursor result, which is the only kind that reports
        # how many rows it touched.
        assert isinstance(result, CursorResult)
        return int(result.rowcount or 0)

    async def existing_content_hashes(self, hashes: Sequence[str]) -> set[str]:
        """Return which of the given content hashes are already stored."""
        wanted = list(dict.fromkeys(hashes))
        found: set[str] = set()
        for start in range(0, len(wanted), HASH_LOOKUP_CHUNK_SIZE):
            window = wanted[start : start + HASH_LOOKUP_CHUNK_SIZE]
            statement = select(LoreChunk.content_hash).where(LoreChunk.content_hash.in_(window))
            result = await self._session.execute(statement)
            found.update(result.scalars().all())
        return found

    @staticmethod
    def _to_model(record: EmbeddedLoreChunk) -> LoreChunk:
        """Translate a corpus record into its ORM representation."""
        return LoreChunk(
            content=record.content,
            content_hash=record.content_hash,
            source_type=str(record.source_type),
            topic=str(record.topic),
            title=record.title,
            character_name=record.character_name,
            source_url=record.source_url,
            embedding=list(record.embedding),
        )
