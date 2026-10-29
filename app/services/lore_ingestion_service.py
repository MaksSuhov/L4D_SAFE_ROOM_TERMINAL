from __future__ import annotations

import hashlib
from collections.abc import Sequence

from app.interfaces import AIClientProtocol, LoreChunkRepositoryProtocol
from app.schemas import EmbeddedLoreChunk, IngestionReport, LoreChunkCreate


def content_hash(content: str) -> str:
    """Return the deduplication hash of a record.

    The text is normalised before hashing so that a re-fetch of the wiki which
    only changes whitespace does not look like new content.
    """
    normalised = " ".join(content.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


class LoreIngestionService:
    """Turns corpus records into embedded :class:`~app.models.LoreChunk` rows.

    The service is idempotent: every record carries a hash of its text and
    records whose hash is already stored are skipped, so re-running the seeder
    neither duplicates rows nor pays for their embeddings again.
    """

    def __init__(
        self,
        lore_chunk_repository: LoreChunkRepositoryProtocol,
        ai_client: AIClientProtocol,
    ) -> None:
        self._lore_chunks = lore_chunk_repository
        self._ai_client = ai_client

    async def ingest(self, items: Sequence[LoreChunkCreate]) -> IngestionReport:
        """Embed and store the records of ``items`` that are not stored yet."""
        if not items:
            return IngestionReport()

        pending = await self._select_pending(items)
        if not pending:
            return IngestionReport(skipped=len(items))

        embeddings = await self._ai_client.get_embeddings([item.content for item in pending])
        records = [
            EmbeddedLoreChunk(
                **item.model_dump(),
                content_hash=content_hash(item.content),
                embedding=embedding,
            )
            for item, embedding in zip(pending, embeddings, strict=True)
        ]
        await self._lore_chunks.create_chunks(records)

        return IngestionReport(
            stored=len(records),
            skipped=len(items) - len(records),
            by_source_type=self._count_by_source_type(records),
        )

    async def _select_pending(self, items: Sequence[LoreChunkCreate]) -> list[LoreChunkCreate]:
        """Drop the records the store already holds, and duplicates within the batch."""
        candidates: dict[str, LoreChunkCreate] = {}
        for item in items:
            candidates.setdefault(content_hash(item.content), item)

        already_stored = await self._lore_chunks.existing_content_hashes(list(candidates))
        return [item for digest, item in candidates.items() if digest not in already_stored]

    @staticmethod
    def _count_by_source_type(records: Sequence[EmbeddedLoreChunk]) -> dict[str, int]:
        """Tally the stored records per evidence tier for the run report."""
        tally: dict[str, int] = {}
        for record in records:
            key = str(record.source_type)
            tally[key] = tally.get(key, 0) + 1
        return tally
