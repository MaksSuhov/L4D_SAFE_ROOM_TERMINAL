"""Двойники в памяти, реализующие порты приложения.

Они доказывают принцип инверсии зависимостей на практике: конвейер RAG
работает на этих заглушках без Qdrant, без модели BM25 и без доступа к сети.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime, timedelta
from typing import Any

from qdrant_client import models

from app.domain import Character, LoreChunk, RefreshSession, User, normalise_email
from app.exceptions import EmailAlreadyRegisteredError
from app.interfaces import ExpandedQuery, RetrievedChunk
from app.schemas import EmbeddedLoreChunk, LoreChunkCreate
from app.taxonomy import LoreTopic, SourceType, UserRole

EMBEDDING_SIZE = 4
"""Заглушкам не важна настоящая ширина вектора эмбеддинга."""


def build_character(
    *,
    character_id: int = 1,
    name: str = "Эллис",
    system_prompt: str = "Ты — Эллис.",
    few_shot_examples: list[dict[str, str]] | None = None,
) -> Character:
    """Создать доменную сущность персонажа без участия хранилища."""
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
    """Создать фрагмент лора с хешем, выведенным из идентификатора.

    Хэш уникален: по нему конвейер отличает одну запись архива от другой, и
    заглушка обязана повторять это свойство.
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
            character.id: character for character in characters or () if character.id is not None
        }

    async def get_by_id(self, entity_id: int) -> Character | None:
        return self._characters.get(entity_id)

    async def get_by_name(self, name: str) -> Character | None:
        for character in self._characters.values():
            if character.name == name:
                return character
        return None

    async def get_all(self) -> Sequence[Character]:
        return sorted(self._characters.values(), key=lambda character: character.id or 0)

    async def save(self, character: Character) -> bool:
        """Записать образ, обновив существующий на месте."""
        existing = await self.get_by_name(character.name)
        if existing is not None:
            existing.system_prompt = character.system_prompt
            existing.few_shot_examples = list(character.few_shot_examples)
            return False
        character.id = max(self._characters, default=0) + 1
        self._characters[character.id] = character
        return True


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
        query_text: str,
        *,
        limit: int,
        source_types: Sequence[SourceType] | None = None,
        topic: str | None = None,
        character_name: str | None = None,
    ) -> Sequence[RetrievedChunk]:
        self.searches.append(
            {
                "query_embedding": list(query_embedding),
                "query_text": query_text,
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
        )


class FakeVectorStore:
    """Хранилище-двойник: проба готовности отвечает, не обращаясь к Qdrant.

    Заменяет только то, о чём спрашивает проба; репозитории рядом получают
    собственные двойники, поэтому клиента и имён коллекций здесь нет.
    """

    def __init__(self, *, reachable: bool = True) -> None:
        self._reachable = reachable

    async def ping(self) -> bool:
        return self._reachable


class FakeSparseEncoder:
    """Разреженный вектор без модели: слово даёт устойчивый индекс, значение — счётчик.

    Двойник нужен там, где проверяется механика гибридного поиска, а не качество
    BM25: он детерминирован, работает офлайн и повторяет главное свойство модели —
    один и тот же термин получает один и тот же индекс и у записи корпуса, и у
    вопроса.
    """

    def __init__(self) -> None:
        self.encoded_documents: list[str] = []
        self.encoded_queries: list[str] = []

    async def encode_documents(self, texts: Sequence[str]) -> list[models.SparseVector]:
        batch = list(texts)
        self.encoded_documents.extend(batch)
        return [self._encode(text) for text in batch]

    async def encode_query(self, text: str) -> models.SparseVector:
        self.encoded_queries.append(text)
        return self._encode(text)

    @staticmethod
    def _encode(text: str) -> models.SparseVector:
        weights: dict[int, float] = {}
        for token in re.findall(r"\w+", text.lower()):
            index = int(hashlib.blake2b(token.encode("utf-8"), digest_size=4).hexdigest(), 16)
            weights[index] = weights.get(index, 0.0) + 1.0
        return models.SparseVector(indices=list(weights), values=list(weights.values()))


class FakeAIClient:
    """Детерминированная заглушка для :class:`~app.services.ai_service.OpenAIClient`."""

    model_name = "fake-model"

    def __init__(
        self,
        *,
        answer: str = "Stay behind the wall, kids.",
        stream_fragments: Sequence[str] | None = None,
        stream_sequences: Sequence[Sequence[str]] | None = None,
        completion: str = "{}",
        stream_errors: Sequence[Exception] | None = None,
    ) -> None:
        self._answer = answer
        self._stream_fragments = list(stream_fragments) if stream_fragments is not None else None
        self._stream_sequences = (
            [list(chunk) for chunk in stream_sequences] if stream_sequences is not None else None
        )
        self._completion = completion
        self._stream_errors = list(stream_errors or ())
        self.embedded_texts: list[str] = []
        self.embedding_calls: list[list[str]] = []
        self.translated_texts: list[str] = []
        self.generation_calls: list[dict[str, object]] = []
        self.completion_calls: list[dict[str, object]] = []
        self.stream_temperatures: list[float | None] = []

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
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        self._record_generation(system_prompt, context, user_message, few_shot_examples)
        self.stream_temperatures.append(temperature)
        if self._stream_errors:
            raise self._stream_errors.pop(0)
        fragments: list[str]
        if self._stream_sequences is not None:
            fragments = self._stream_sequences.pop(0) if self._stream_sequences else []
        else:
            fragments = self._stream_fragments or [self._answer]
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


class FakeSimilarity:
    """Близость ответа и эталона, заданная числом: без модели эмбеддингов."""

    def __init__(self, value: float = 0.0) -> None:
        self._value = value
        self.calls: list[tuple[str, str]] = []

    async def similarity(self, reference: str, answer: str) -> float:
        self.calls.append((reference, answer))
        return self._value


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


class RecordingSpan:
    """Участок трейсера-двойника: запоминает, чем его разметили."""

    def __init__(self) -> None:
        self.annotations: list[dict[str, Any]] = []

    def annotate(
        self,
        *,
        input: Any = None,
        output: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Запомнить разметку участка."""
        self.annotations.append({"input": input, "output": output, "metadata": metadata})

    @property
    def output(self) -> Any:
        """Выход последней разметки, если она была."""
        return self.annotations[-1]["output"] if self.annotations else None

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Метаданные последней разметки; пустой словарь, если разметки не было."""
        if not self.annotations:
            return {}
        last: Mapping[str, Any] | None = self.annotations[-1]["metadata"]
        return last or {}


class RecordingTracer:
    """Трейсер-двойник: запоминает открытые участки, чтобы тест их проверил.

    Он доказывает, что сервисы зависят от порта, а не от SDK наблюдаемости:
    конвейер размечается ровно так же, как с настоящим трейсером, но без сети.
    """

    def __init__(self) -> None:
        self.spans: list[tuple[str, str, RecordingSpan]] = []

    def span(
        self,
        name: str,
        *,
        as_type: str = "span",
    ) -> AbstractContextManager[RecordingSpan]:
        """Открыть участок и запомнить его имя вместе с типом."""
        span = RecordingSpan()
        self.spans.append((name, as_type, span))
        return nullcontext(span)

    def flush(self) -> None:
        """Двойнику нечего отправлять."""

    def shutdown(self) -> None:
        """Двойнику нечего освобождать."""

    @property
    def names(self) -> list[str]:
        """Имена открытых участков в порядке открытия."""
        return [name for name, _, _ in self.spans]

    def named(self, name: str) -> RecordingSpan:
        """Вернуть последний открытый участок с таким именем."""
        for span_name, _, span in reversed(self.spans):
            if span_name == name:
                return span
        raise AssertionError(f"no span named {name!r}")


def build_user(
    *,
    user_id: str = "user-1",
    email: str = "ellis@l4d.dev",
    password_hash: str = "fake$correct-horse-battery",
    display_name: str = "Эллис",
    role: UserRole = UserRole.MEMBER,
    is_active: bool = True,
) -> User:
    """Создать учётную запись без участия хранилища."""
    return User(
        email=email,
        password_hash=password_hash,
        display_name=display_name,
        role=role,
        is_active=is_active,
        id=user_id,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


class FixedClock:
    """Часы, которые стоят, пока их не сдвинут.

    Сроки жизни токенов и сессий проверяются ожиданием в реальном времени — это
    медленно и ненадёжно, поэтому тесты двигают часы сами.
    """

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        """Перевести часы вперёд на указанное число секунд."""
        self._now += timedelta(seconds=seconds)


class FakePasswordHasher:
    """Хеширование без Argon2: тестам важна логика доступа, а не стоимость перебора.

    Настоящий хешер проверяется отдельным тестом на настоящем Argon2; здесь он
    стоял бы десятки миллисекунд на каждый вызов и ничего бы не проверял.
    """

    def __init__(self) -> None:
        self.hashed: list[str] = []
        self.verified: list[str] = []
        self.equalised: list[str] = []
        self.stale: set[str] = set()
        """Хеши, которые двойник объявит устаревшими, чтобы проверить их пересчёт."""

    def hash(self, password: str) -> str:
        self.hashed.append(password)
        return f"fake${password}"

    def verify(self, password: str, encoded: str) -> bool:
        self.verified.append(password)
        return encoded == f"fake${password}"

    def needs_rehash(self, encoded: str) -> bool:
        return encoded in self.stale

    def equalise_timing(self, password: str) -> None:
        self.equalised.append(password)


class InMemoryUserRepository:
    """Реализация :class:`UserRepositoryProtocol` на словаре."""

    def __init__(self, users: Sequence[User] | None = None) -> None:
        self.users: dict[str, User] = {normalise_email(user.email): user for user in users or ()}
        self._issued = 0

    async def get_by_id(self, entity_id: str) -> User | None:
        return next((user for user in self.users.values() if user.id == entity_id), None)

    async def get_by_email(self, email: str) -> User | None:
        return self.users.get(normalise_email(email))

    async def get_all(self) -> Sequence[User]:
        return sorted(self.users.values(), key=lambda user: user.email)

    async def create(self, user: User) -> None:
        key = normalise_email(user.email)
        if key in self.users:
            raise EmailAlreadyRegisteredError(key)
        if user.id is None:
            self._issued += 1
            user.id = f"user-{self._issued}"
        self.users[key] = user

    async def update(self, user: User) -> None:
        self.users[normalise_email(user.email)] = user

    async def delete(self, entity_id: str) -> bool:
        found = await self.get_by_id(entity_id)
        if found is None:
            return False
        del self.users[normalise_email(found.email)]
        return True

    async def count(self) -> int:
        return len(self.users)


class InMemoryRefreshSessionRepository:
    """Реализация :class:`RefreshSessionRepositoryProtocol` на списке."""

    def __init__(self) -> None:
        self.sessions: list[RefreshSession] = []
        self._issued = 0

    async def create(self, session: RefreshSession) -> None:
        self._issued += 1
        session.id = f"session-{self._issued}"
        self.sessions.append(session)

    async def get_by_digest(self, digest: str) -> RefreshSession | None:
        return next(
            (session for session in self.sessions if session.token_digest == digest),
            None,
        )

    async def revoke(
        self,
        session_id: str,
        *,
        revoked_at: datetime,
        replaced_by: str | None = None,
    ) -> None:
        for session in self.sessions:
            if session.id == session_id:
                session.revoked_at = revoked_at
                session.replaced_by = replaced_by

    async def revoke_family(self, family_id: str, *, revoked_at: datetime) -> int:
        return await self._revoke(
            [session for session in self.sessions if session.family_id == family_id],
            revoked_at,
        )

    async def revoke_all_for_user(self, user_id: str, *, revoked_at: datetime) -> int:
        return await self._revoke(
            [session for session in self.sessions if session.user_id == user_id],
            revoked_at,
        )

    async def _revoke(self, candidates: list[RefreshSession], revoked_at: datetime) -> int:
        """Погасить перечисленные сессии и посчитать, сколько было действующих."""
        active = [session for session in candidates if not session.is_revoked]
        for session in active:
            session.revoked_at = revoked_at
        return len(active)

    @property
    def active(self) -> list[RefreshSession]:
        """Действующие сессии: те, по которым ещё можно получить новую пару."""
        return [session for session in self.sessions if not session.is_revoked]
