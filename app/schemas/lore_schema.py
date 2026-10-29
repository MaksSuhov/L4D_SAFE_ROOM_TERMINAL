"""Schemas describing the lore archive contract."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoreChunkCreate(BaseModel):
    """A lore fragment waiting to be embedded and stored."""

    content: str = Field(min_length=1)
    data_type: str = Field(min_length=1, max_length=64)


class LoreSource(BaseModel):
    """A retrieved chunk returned alongside a generated answer."""

    id: int
    data_type: str
    content: str
    similarity: float = Field(ge=-1.0, le=1.0)
