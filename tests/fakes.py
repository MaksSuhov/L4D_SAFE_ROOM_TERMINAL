"""Двойники в памяти, реализующие порты приложения.

Они доказывают принцип инверсии зависимостей на практике: конвейер RAG
работает на этих заглушках без PostgreSQL, pgvector и доступа к сети.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence

from app.interfaces import ExpandedQuery, RetrievedChunk
from app.models import Character, LoreChunk
from app.schemas import EmbeddedLoreChunk, LoreChunkCreate
from app.taxonomy import LoreTopic, SourceType

EMBEDDING_SIZE = 4
"""Заглушкам не важна настоящая ширина вектора эмбеддинга."""


def build_character(
    *,
    character_id: int = 1,
    name: str = "Эллис",
    system_prompt: str = "Ты — Эллис.",
    few_shot_examples: list[dict[str, str]] | None = None,
) -> Character:
    """Создать отсоединённую ORM-сущность персонажа, сессия не нужна."""
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
    """Создать отсоединённый фрагмент лора с фиктивным эмбеддингом.

    Хэш выводится из идентификатора и потому уникален: по нему конвейер отличает
    одну запись архива от другой, и заглушка обязана повторять это свойство.
    """
    return LoreChunk(
        id=chunk_id,
        content=content,
        content_hash=f"{chunk_id:064x}",
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
    """Создать запись корпуса, ожидающую загрузки."""
    return LoreChunkCreate(
        content=content,
        source_type=source_type,
        topic=topic,
        title=title,
        character_name=character_name,
    )


class InMemoryCharacterRepository:
    """Реализация :class:`CharacterRepositoryProtocol` на словаре."""

    def __init__(self, characters: Sequence[Character] | None = None) -> None:
        self._characters: dict[int, Character] = {
            character.id: character for character in characters or ()
        }

    async def get_by_id(self, entity_id: int) -> Character | None:
        return self._characters.get(entity_id)

    async def get_all(self) -> Sequence[Character]:
        return sorted(self._characters.values(), key=lambda character: character.id)


class InMemoryLoreChunkRepository:
    """Отдаёт заранее заданный упорядоченный список совпадений и записывает каждый поиск.

    Заданный список заменяет векторный индекс: настоящий репозиторий возвращает
    совпадения, упорядоченные по косинусной дистанции, и этот двойник делает так же,
    что и позволяет тестам уровней проверять бюджеты и фильтры без базы.

    ``matches_of`` нужен там, где выдача обязана зависеть от формулировки: он получает
    вектор запроса и отдаёт кандидатов именно для него, — так проверяется поиск по
    нескольким формулировкам HIDE. Фильтры уровня при этом продолжают действовать.
    """

    def __init__(
        self,
        matches: Sequence[RetrievedChunk] | None = None,
        *,
        matches_of: Callable[[Sequence[float]], Sequence[RetrievedChunk]] | None = None,
    ) -> None:
        self._matches: list[RetrievedChunk] = list(matches or ())
        self._matches_of = matches_of
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
        candidates = (
            self._matches_of(query_embedding) if self._matches_of is not None else self._matches
        )
        eligible = [
            match
            for match in candidates
            if self._matches_filters(match, source_types, topic, character_name)
        ]
        return eligible[:limit]

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
    """Детерминированная заглушка для :class:`~app.services.ai_service.OpenAIClient`."""

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
        self.embedding_calls: list[list[str]] = []
        self.translated_texts: list[str] = []
        self.generation_calls: list[dict[str, object]] = []
        self.completion_calls: list[dict[str, object]] = []

    async def get_embedding(self, text: str) -> list[float]:
        return (await self.get_embeddings([text]))[0]

    async def get_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        self.embedding_calls.append(list(texts))
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


class FakeQueryExpander:
    """Отдаёт заранее заданные формулировки и не обращается к провайдеру."""

    def __init__(self, hypotheticals: Sequence[str] | None = None) -> None:
        self._hypotheticals = tuple(hypotheticals or ())
        self.questions: list[str] = []

    async def expand(self, question: str) -> ExpandedQuery:
        self.questions.append(question)
        return ExpandedQuery(original=question, hypotheticals=self._hypotheticals)


class FakeClock:
    """Часы, прибавляющие фиксированный шаг при каждом чтении.

    Они позволяют тестам замеров проверять точные числа без ожидания: при шаге в
    одну секунду замеры в начале, на первом токене и в конце дают одну секунду
    ожидания первого токена и две секунды всего.
    """

    def __init__(self, *, start: float = 0.0, step: float = 0.0) -> None:
        self._now = start
        self._step = step

    def now(self) -> float:
        current = self._now
        self._now += self._step
        return current
