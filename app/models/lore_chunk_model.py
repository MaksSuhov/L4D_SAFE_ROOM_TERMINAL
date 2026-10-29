"""ORM model of a retrievable lore fragment with its pgvector embedding."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, DateTime, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.config import get_settings
from app.database import Base
from app.taxonomy import SourceType

EMBEDDING_DIMENSIONS: Final[int] = get_settings().embedding_dimensions
"""Vector width of the embedding column; must match the configured embedding model."""

SOURCE_TYPE_VALUES: Final[tuple[str, ...]] = tuple(member.value for member in SourceType)
"""Allowed values of ``source_type``, derived from the taxonomy so the database
constraint can never drift away from the enum."""

_CONTENT_HASH_LENGTH: Final[int] = 64
"""Length of a SHA-256 hex digest, kept in sync with the hashing helper."""


class LoreChunk(Base):
    """A single retrievable fragment of Left 4 Dead lore with its embedding.

    Besides the text and the vector, every row carries its provenance — where it
    came from and how much it can be trusted — because the retrieval pipeline
    answers differently depending on :attr:`source_type`.
    """

    __tablename__ = "lore_chunks"
    __table_args__ = (
        CheckConstraint(
            f"source_type IN ({', '.join(repr(value) for value in SOURCE_TYPE_VALUES)})",
            name="ck_lore_chunks_source_type",
        ),
        Index(
            "ix_lore_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # Retrieval always filters by tier first, so the tier leads the index and
        # the topic follows it for the narrow-by-subject queries.
        Index("ix_lore_chunks_source_type_topic", "source_type", "topic"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(
        String(_CONTENT_HASH_LENGTH),
        nullable=False,
        unique=True,
        index=True,
    )
    source_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    topic: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    character_name: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    source_url: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        preview: str = self.content[:40]
        return (
            f"LoreChunk(id={self.id!r}, source_type={self.source_type!r}, "
            f"title={self.title!r}, content={preview!r})"
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable view without the embedding vector."""
        return {
            "id": self.id,
            "content": self.content,
            "content_hash": self.content_hash,
            "source_type": self.source_type,
            "topic": self.topic,
            "title": self.title,
            "character_name": self.character_name,
            "source_url": self.source_url,
            "created_at": self.created_at,
        }
