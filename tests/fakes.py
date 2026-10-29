"""In-memory doubles implementing the application ports.

They prove the Dependency Inversion Principle in practice: the RAG pipeline runs
against these fakes without PostgreSQL, pgvector or network access.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.interfaces import RetrievedChunk
from app.models import Character, LoreChunk

EMBEDDING_SIZE = 4
"""The fakes do not care about the real width of an embedding vector."""


def build_character(
    *,
    character_id: int = 1,
    name: str = "Ellis",
    system_prompt: str = "You are Ellis.",
    few_shot_examples: list[dict[str, str]] | None = None,
) -> Character:
    """Create a detached ORM character, no session required."""
    return Character(
        id=character_id,
        name=name,
        system_prompt=system_prompt,
        few_shot_examples=few_shot_examples if few_shot_examples is not None else [],
    )


def build_chunk(
    *,
    chunk_id: int,
    content: str,
    data_type: str = "monster",
) -> LoreChunk:
    """Create a detached ORM lore chunk with a dummy embedding."""
    return LoreChunk(
        id=chunk_id,
        content=content,
        data_type=data_type,
        embedding=[0.1] * EMBEDDING_SIZE,
    )


class InMemoryCharacterRepository:
    """Dictionary-backed implementation of :class:`CharacterRepositoryProtocol`."""

    def __init__(self, characters: Sequence[Character] | None = None) -> None:
        self._characters: dict[int, Character] = {
            character.id: character for character in characters or ()
        }

    async def get_by_id(self, entity_id: int) -> Character | None:
        return self._characters.get(entity_id)

    async def get_all(self) -> Sequence[Character]:
        return sorted(self._characters.values(), key=lambda character: character.id)


class InMemoryLoreChunkRepository:
    """Returns a canned, ordered list of matches and records every search."""

    def __init__(self, matches: Sequence[RetrievedChunk] | None = None) -> None:
        self._matches: list[RetrievedChunk] = list(matches or ())
        self.searches: list[dict[str, object]] = []
        self.stored: list[LoreChunk] = []

    async def search_similar_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        data_type: str | None = None,
    ) -> Sequence[RetrievedChunk]:
        self.searches.append(
            {"query_embedding": list(query_embedding), "limit": limit, "data_type": data_type}
        )
        candidates = [
            match
            for match in self._matches
            if data_type is None or match.chunk.data_type == data_type
        ]
        return candidates[:limit]

    async def create_chunk(
        self,
        *,
        content: str,
        data_type: str,
        embedding: Sequence[float],
    ) -> LoreChunk:
        chunk = build_chunk(chunk_id=len(self.stored) + 1, content=content, data_type=data_type)
        self.stored.append(chunk)
        return chunk

    async def create_chunks(
        self,
        *,
        contents: Sequence[str],
        data_types: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> Sequence[LoreChunk]:
        return [
            await self.create_chunk(content=content, data_type=data_type, embedding=embedding)
            for content, data_type, embedding in zip(
                contents, data_types, embeddings, strict=True
            )
        ]

    async def count(self) -> int:
        return len(self.stored)


class FakeAIClient:
    """Deterministic stand-in for :class:`~app.services.ai_service.OpenAIClient`."""

    model_name = "fake-model"

    def __init__(self, *, answer: str = "Stay behind the wall, kids.") -> None:
        self._answer = answer
        self.embedded_texts: list[str] = []
        self.generation_calls: list[dict[str, object]] = []

    async def get_embedding(self, text: str) -> list[float]:
        self.embedded_texts.append(text)
        return [float(len(text))] * EMBEDDING_SIZE

    async def generate_response(
        self,
        system_prompt: str,
        context: str,
        user_message: str,
        few_shot_examples: Sequence[Mapping[str, str]] | None = None,
    ) -> str:
        self.generation_calls.append(
            {
                "system_prompt": system_prompt,
                "context": context,
                "user_message": user_message,
                "few_shot_examples": list(few_shot_examples or ()),
            }
        )
        return self._answer
