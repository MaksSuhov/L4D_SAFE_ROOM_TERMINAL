"""In-memory doubles implementing the application ports.

They prove the Dependency Inversion Principle in practice: the RAG pipeline runs
against these fakes without PostgreSQL, pgvector or network access.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence

from app.interfaces import RetrievedChunk
from app.models import Character, LoreChunk
from app.schemas import EmbeddedLoreChunk, LoreChunkCreate
from app.taxonomy import LoreTopic, SourceType

EMBEDDING_SIZE = 4
"""The fakes do not care about the real width of an embedding vector."""


def build_character(
    *,
    character_id: int = 1,
    name: str = "Эллис",
    system_prompt: str = "Ты — Эллис.",
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
    source_type: SourceType = SourceType.OFFICIAL,
    topic: LoreTopic = LoreTopic.INFECTED,
    title: str = "Archive",
    character_name: str | None = None,
) -> LoreChunk:
    """Create a detached ORM lore chunk with a dummy embedding."""
    return LoreChunk(
        id=chunk_id,
        content=content,
        content_hash=f"hash-{chunk_id:064d}"[:64],
        source_type=str(source_type),
        topic=str(topic),
        title=title,
        character_name=character_name,
        source_url="",
        embedding=[0.1] * EMBEDDING_SIZE,
    )


def build_document(
    *,
    content: str,
    source_type: SourceType = SourceType.OFFICIAL,
    topic: LoreTopic = LoreTopic.INFECTED,
    title: str = "Archive",
    character_name: str | None = None,
) -> LoreChunkCreate:
    """Create a corpus record waiting to be ingested."""
    return LoreChunkCreate(
        content=content,
        source_type=source_type,
        topic=topic,
        title=title,
        character_name=character_name,
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
    """Serves a canned, ordered list of matches and records every search.

    The canned list stands in for the vector index: the real repository returns
    its matches ordered by cosine distance, and this double does the same, which
    is what lets the tier tests assert on budgets and filters without a database.
    """

    def __init__(self, matches: Sequence[RetrievedChunk] | None = None) -> None:
        self._matches: list[RetrievedChunk] = list(matches or ())
        self.searches: list[dict[str, object]] = []
        self.stored: list[LoreChunk] = []

    async def search_similar_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        source_types: Sequence[SourceType] | None = None,
        topic: str | None = None,
        character_name: str | None = None,
    ) -> Sequence[RetrievedChunk]:
        self.searches.append(
            {
                "query_embedding": list(query_embedding),
                "limit": limit,
                "source_types": list(source_types) if source_types is not None else None,
                "topic": topic,
                "character_name": character_name,
            }
        )
        candidates = [
            match
            for match in self._matches
            if self._matches_filters(match, source_types, topic, character_name)
        ]
        return candidates[:limit]

    async def create_chunk(self, record: EmbeddedLoreChunk) -> LoreChunk:
        chunk = self._to_chunk(len(self.stored) + 1, record)
        self.stored.append(chunk)
        return chunk

    async def create_chunks(self, records: Sequence[EmbeddedLoreChunk]) -> Sequence[LoreChunk]:
        return [await self.create_chunk(record) for record in records]

    async def existing_content_hashes(self, hashes: Sequence[str]) -> set[str]:
        stored = {chunk.content_hash for chunk in self.stored}
        return {digest for digest in hashes if digest in stored}

    async def count(self) -> int:
        return len(self._matches) + len(self.stored)

    @staticmethod
    def _matches_filters(
        match: RetrievedChunk,
        source_types: Sequence[SourceType] | None,
        topic: str | None,
        character_name: str | None,
    ) -> bool:
        chunk = match.chunk
        if source_types is not None and chunk.source_type not in {
            str(source_type) for source_type in source_types
        }:
            return False
        if topic is not None and chunk.topic != topic:
            return False
        return character_name is None or chunk.character_name == character_name

    @staticmethod
    def _to_chunk(chunk_id: int, record: EmbeddedLoreChunk) -> LoreChunk:
        return LoreChunk(
            id=chunk_id,
            content=record.content,
            content_hash=record.content_hash,
            source_type=str(record.source_type),
            topic=str(record.topic),
            title=record.title,
            character_name=record.character_name,
            source_url=record.source_url,
            embedding=list(record.embedding),
        )


class FakeAIClient:
    """Deterministic stand-in for :class:`~app.services.ai_service.OpenAIClient`."""

    model_name = "fake-model"

    def __init__(
        self,
        *,
        answer: str = "Stay behind the wall, kids.",
        stream_fragments: Sequence[str] | None = None,
        completion: str = "{}",
    ) -> None:
        self._answer = answer
        self._stream_fragments = list(stream_fragments) if stream_fragments is not None else None
        self._completion = completion
        self.embedded_texts: list[str] = []
        self.translated_texts: list[str] = []
        self.generation_calls: list[dict[str, object]] = []
        self.completion_calls: list[dict[str, object]] = []

    async def get_embedding(self, text: str) -> list[float]:
        return (await self.get_embeddings([text]))[0]

    async def get_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return [[float(len(text))] * EMBEDDING_SIZE for text in texts]

    async def translate_texts(self, texts: Sequence[str]) -> list[str]:
        self.translated_texts.extend(texts)
        return [f"[ru] {text}" for text in texts]

    async def generate_response(
        self,
        system_prompt: str,
        context: str,
        user_message: str,
        few_shot_examples: Sequence[Mapping[str, str]] | None = None,
    ) -> str:
        self._record_generation(system_prompt, context, user_message, few_shot_examples)
        return self._answer

    async def stream_response(
        self,
        system_prompt: str,
        context: str,
        user_message: str,
        few_shot_examples: Sequence[Mapping[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        self._record_generation(system_prompt, context, user_message, few_shot_examples)
        fragments = self._stream_fragments
        if fragments is None:
            fragments = [self._answer]
        for fragment in fragments:
            yield fragment

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.0,
    ) -> str:
        self.completion_calls.append(
            {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "temperature": temperature,
            }
        )
        return self._completion

    def _record_generation(
        self,
        system_prompt: str,
        context: str,
        user_message: str,
        few_shot_examples: Sequence[Mapping[str, str]] | None,
    ) -> None:
        self.generation_calls.append(
            {
                "system_prompt": system_prompt,
                "context": context,
                "user_message": user_message,
                "few_shot_examples": list(few_shot_examples or ()),
            }
        )


class FakeClock:
    """Clock that advances by a fixed step on every reading.

    It lets the latency tests assert on exact numbers without sleeping: with a
    step of one second, a reading taken at the start, on the first token and at
    the end reports a one-second wait to the first token and two seconds total.
    """

    def __init__(self, *, start: float = 0.0, step: float = 0.0) -> None:
        self._now = start
        self._step = step

    def now(self) -> float:
        current = self._now
        self._now += self._step
        return current
