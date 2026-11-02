"""Порты приложения (принцип инверсии зависимостей).

Слой сервисов зависит от этих определений ``Protocol``, а не от конкретных
репозиториев SQLAlchemy или обёртки над OpenAI SDK. Вместо них можно внедрить
любой объект подходящей формы — настоящий адаптер, заглушку из тестов или
будущий клиент векторной базы.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

from app.models import Character, LoreChunk
from app.schemas import EmbeddedLoreChunk, LoreSource
from app.taxonomy import SourceType


class RetrievedChunk(NamedTuple):
    """Фрагмент лора вместе с косинусной близостью поискового запроса."""

    chunk: LoreChunk
    similarity: float


@dataclass(frozen=True, slots=True)
class PreparedChat:
    """Всё, что нужно одному ходу чата после выполнения выборки.

    Это стык между двумя половинами конвейера: выборка его заполняет
    (вопрос -> доказательства -> брифинг), а генерации нужно только то, что здесь
    хранится. Открыв его наружу, вызывающий код может стримить ответ и сохранять
    доказательства — именно так и поступает офлайн-стенд оценки.
    """

    character_id: int
    character_name: str
    model: str
    system_prompt: str
    context: str
    few_shot_examples: Sequence[Mapping[str, str]]
    sources: Sequence[LoreSource]


class CharacterRepositoryProtocol(Protocol):
    """Доступ на чтение к образам персонажей."""

    async def get_by_id(self, entity_id: int) -> Character | None:
        """Вернуть персонажа с данным идентификатором или ``None``."""
        ...

    async def get_all(self) -> Sequence[Character]:
        """Вернуть всех сохранённых персонажей, упорядоченных по идентификатору."""
        ...


class LoreChunkRepositoryProtocol(Protocol):
    """Сохранение и векторный поиск по фрагментам лора."""

    async def create_chunk(self, record: EmbeddedLoreChunk) -> LoreChunk:
        """Сохранить один проиндексированный фрагмент лора."""
        ...

    async def create_chunks(self, records: Sequence[EmbeddedLoreChunk]) -> Sequence[LoreChunk]:
        """Сохранить несколько проиндексированных фрагментов лора за один запрос."""
        ...

    async def search_similar_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        source_types: Sequence[SourceType] | None = None,
        topic: str | None = None,
        character_name: str | None = None,
    ) -> Sequence[RetrievedChunk]:
        """Вернуть ``limit`` ближайших фрагментов по убыванию близости.

        ``source_types`` ограничивает поиск одним или несколькими уровнями
        доказательств — именно это делает двухуровневую выборку возможной: факты
        ищутся среди официальных записей, а слухи — среди надписей на стенах.
        """
        ...

    async def existing_content_hashes(self, hashes: Sequence[str]) -> set[str]:
        """Вернуть, какие из данных хешей содержимого уже сохранены."""
        ...

    async def count(self) -> int:
        """Вернуть число сохранённых фрагментов лора."""
        ...


class AIClientProtocol(Protocol):
    """Доступ к провайдеру языковой модели."""

    model_name: str
    """Имя чат-модели; показывается в ответах API."""

    async def get_embedding(self, text: str) -> list[float]:
        """Посчитать эмбеддинг ``text`` в вектор из ``embedding_dimensions`` чисел."""
        ...

    async def get_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """Посчитать эмбеддинги пакета текстов, сохранив их порядок.

        Пакетность важна: корпус содержит сотни записей, и запрос на каждую запись
        превратил бы наполнение базы в перерыв на кофе.
        """
        ...

    async def translate_texts(self, texts: Sequence[str]) -> list[str]:
        """Перевести пакет текстов на заданный язык корпуса.

        Используется один раз, пока строится корпус, так что английская вики
        превращается в корпус на том языке, на котором отвечает терминал. Реализации
        обязаны вернуть ровно один перевод на вход, откатываясь к исходному тексту
        вместо того, чтобы выбросить запись.
        """
        ...

    async def generate_response(
        self,
        system_prompt: str,
        context: str,
        user_message: str,
        few_shot_examples: Sequence[Mapping[str, str]] | None = None,
    ) -> str:
        """Сгенерировать ответ ассистента, опираясь на ``context``."""
        ...

    def stream_response(
        self,
        system_prompt: str,
        context: str,
        user_message: str,
        few_shot_examples: Sequence[Mapping[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """Отдавать ответ по фрагментам, в том порядке, в котором их шлёт провайдер.

        Тот же ответ, что и у :meth:`generate_response`, только доставляемый потоком
        фрагментов. Стриминг делает наблюдаемым время до первого токена: пока не
        пришёл первый фрагмент, ничего ещё не сгенерировано.
        """
        ...

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.0,
    ) -> str:
        """Ответить на одну инструкцию, без выборки и без персонажей.

        Низкоуровневая точка входа для задач, которым нужен только ответ модели:
        оценить ответ, классифицировать текст, извлечь поле. Температура здесь
        аргумент, потому что такие задачи хотят детерминизма, а не творчества.
        """
        ...
