"""Schemas describing the lore archive contract.

Three shapes travel through the pipeline:

* :class:`LoreChunkCreate` — a normalised corpus record, before it is embedded.
  It is also the row format of the on-disk JSONL corpus snapshot.
* :class:`EmbeddedLoreChunk` — the same record with its vector and content hash,
  which is what the repository persists.
* :class:`LoreSource` — the evidence handed back to an API client, annotated
  with the cosine similarity that retrieved it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.taxonomy import LoreTopic, SourceType

CONTENT_HASH_LENGTH = 64
"""SHA-256 hex digest length, matching the ``content_hash`` column."""


class LoreChunkCreate(BaseModel):
    """A lore fragment waiting to be embedded and stored."""

    content: str = Field(min_length=1)
    source_type: SourceType = Field(
        description="Evidence tier: verified lore, wall graffiti or a character voice sample.",
    )
    topic: LoreTopic = Field(description="Wiki section the record was extracted from.")
    title: str = Field(
        min_length=1,
        max_length=256,
        description="Human-readable origin, e.g. 'No Mercy › The Subway'.",
    )
    character_name: str | None = Field(
        default=None,
        description="Speaker the record belongs to; only set for voice samples.",
    )
    source_url: str = Field(default="", max_length=512)


class EmbeddedLoreChunk(LoreChunkCreate):
    """A corpus record together with its embedding and deduplication hash."""

    content_hash: str = Field(min_length=CONTENT_HASH_LENGTH, max_length=CONTENT_HASH_LENGTH)
    embedding: list[float] = Field(min_length=1)


class LoreSource(BaseModel):
    """A retrieved chunk returned alongside a generated answer."""

    id: int
    source_type: SourceType
    topic: LoreTopic
    title: str
    character_name: str | None = None
    content: str
    similarity: float = Field(ge=-1.0, le=1.0)


class IngestionReport(BaseModel):
    """Outcome of one ingestion run, surfaced by the CLI and by tests."""

    stored: int = 0
    skipped: int = 0
    by_source_type: dict[str, int] = Field(default_factory=dict)

    @property
    def total(self) -> int:
        """Number of records offered to the run, stored or skipped."""
        return self.stored + self.skipped
