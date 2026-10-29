"""ORM model of a retrievable lore fragment with its pgvector embedding."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.config import get_settings
from app.database import Base

EMBEDDING_DIMENSIONS: Final[int] = get_settings().embedding_dimensions
"""Vector width of the embedding column; must match the configured embedding model."""


class LoreChunk(Base):
    """A single retrievable fragment of Left 4 Dead lore with its embedding."""

    __tablename__ = "lore_chunks"
    __table_args__ = (
        Index(
            "ix_lore_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    data_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
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
        return f"LoreChunk(id={self.id!r}, data_type={self.data_type!r}, content={preview!r})"

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable view without the embedding vector."""
        return {
            "id": self.id,
            "content": self.content,
            "data_type": self.data_type,
            "created_at": self.created_at,
        }
