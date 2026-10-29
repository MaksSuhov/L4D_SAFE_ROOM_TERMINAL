"""Ports of the application (Dependency Inversion Principle).

The service layer depends on these ``Protocol`` definitions rather than on the
concrete SQLAlchemy repositories or on the OpenAI SDK wrapper. Any object with a
matching shape — a real adapter, a fake used in tests, or a future vector
database client — can be injected instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple, Protocol

from app.models import Character, LoreChunk


class RetrievedChunk(NamedTuple):
    """A lore chunk paired with the cosine similarity of a search query."""

    chunk: LoreChunk
    similarity: float


class CharacterRepositoryProtocol(Protocol):
    """Read access to character personas."""

    async def get_by_id(self, entity_id: int) -> Character | None:
        """Return the character with the given identifier, or ``None``."""
        ...

    async def get_all(self) -> Sequence[Character]:
        """Return every stored character ordered by identifier."""
        ...


class LoreChunkRepositoryProtocol(Protocol):
    """Persistence and vector search over lore fragments."""

    async def create_chunk(
        self,
        *,
        content: str,
        data_type: str,
        embedding: Sequence[float],
    ) -> LoreChunk:
        """Persist a single embedded lore fragment."""
        ...

    async def create_chunks(
        self,
        *,
        contents: Sequence[str],
        data_types: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> Sequence[LoreChunk]:
        """Persist several embedded lore fragments in one round trip."""
        ...

    async def search_similar_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        data_type: str | None = None,
    ) -> Sequence[RetrievedChunk]:
        """Return the ``limit`` closest chunks ordered by descending similarity."""
        ...

    async def count(self) -> int:
        """Return the number of stored lore fragments."""
        ...


class AIClientProtocol(Protocol):
    """Access to the language model provider."""

    model_name: str
    """Name of the chat model, surfaced in API responses."""

    async def get_embedding(self, text: str) -> list[float]:
        """Embed ``text`` into a vector of ``embedding_dimensions`` floats."""
        ...

    async def generate_response(
        self,
        system_prompt: str,
        context: str,
        user_message: str,
        few_shot_examples: Sequence[Mapping[str, str]] | None = None,
    ) -> str:
        """Produce the assistant answer grounded in ``context``."""
        ...
