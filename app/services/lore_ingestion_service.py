"""Lore ingestion: embedding raw archive text into the vector store."""

from __future__ import annotations

from collections.abc import Sequence

from app.interfaces import AIClientProtocol, LoreChunkRepositoryProtocol
from app.schemas import LoreChunkCreate


class LoreIngestionService:
    """Turns raw lore text into embedded :class:`~app.models.LoreChunk` rows."""

    def __init__(
        self,
        lore_chunk_repository: LoreChunkRepositoryProtocol,
        ai_client: AIClientProtocol,
    ) -> None:
        self._lore_chunks = lore_chunk_repository
        self._ai_client = ai_client

    async def ingest(self, items: Sequence[LoreChunkCreate]) -> int:
        """Embed and store ``items``; returns the number of stored chunks.

        Embeddings are requested sequentially on purpose: the corpus is small and
        a bounded, predictable request rate keeps the provider happy.
        """
        if not items:
            return 0

        embeddings: list[list[float]] = []
        for item in items:
            embeddings.append(await self._ai_client.get_embedding(item.content))

        stored = await self._lore_chunks.create_chunks(
            contents=[item.content for item in items],
            data_types=[item.data_type for item in items],
            embeddings=embeddings,
        )
        return len(stored)
