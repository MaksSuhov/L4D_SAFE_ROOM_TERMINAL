"""Pydantic schemas: the contract between the API layer and its clients.

ORM models never leave the service layer, they are converted into these
schemas, which keeps the HTTP contract stable even if the persistence model
changes. The package facade re-exports every schema so callers keep a single
import path while the definitions are grouped by aggregate.
"""

from app.schemas.character_schema import (
    CharacterBase,
    CharacterCreate,
    CharacterRead,
    FewShotExample,
)
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.schemas.health_schema import HealthResponse
from app.schemas.lore_schema import (
    EmbeddedLoreChunk,
    IngestionReport,
    LoreChunkCreate,
    LoreSource,
)

__all__ = [
    "CharacterBase",
    "CharacterCreate",
    "CharacterRead",
    "ChatRequest",
    "ChatResponse",
    "EmbeddedLoreChunk",
    "FewShotExample",
    "HealthResponse",
    "IngestionReport",
    "LoreChunkCreate",
    "LoreSource",
]
