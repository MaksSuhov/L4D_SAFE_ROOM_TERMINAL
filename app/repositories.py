"""Слой репозиториев: единственное место, которое знает SQL.

Репозитории прячут построение запросов за методами с говорящими именами, так
что слой сервисов никогда не трогает ``select()`` или API сессии напрямую.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, TypeVar

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Base
from app.interfaces import (
    CharacterRepositoryProtocol,
    LoreChunkRepositoryProtocol,
    RetrievedChunk,
)
from app.models import Character, LoreChunk
from app.schemas import EmbeddedLoreChunk
from app.taxonomy import SourceType

ModelT = TypeVar("ModelT", bound=Base)

HASH_LOOKUP_CHUNK_SIZE = 1000
"""Верхняя граница числа параметров одной выборки ``IN (...)``: большой корпус
никогда не породит запрос, который сервер откажется планировать.
"""


class BaseRepository(Generic[ModelT]):
    """Поведение сохранения, общее для каждого репозитория одной модели."""

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @property
    def session(self) -> AsyncSession:
        """Единица работы, в которой участвует этот репозиторий."""
        return self._session

    async def get_by_id(self, entity_id: int) -> ModelT | None:
        """Получить сущность по первичному ключу, по возможности через карту идентичности."""
        return await self._session.get(self.model, entity_id)

    async def add(self, instance: ModelT) -> ModelT:
        """Сохранить ``instance`` и обновить его, чтобы заполнить значения по умолчанию."""
        self._session.add(instance)
        await self._session.flush()
        await self._session.refresh(instance)
        return instance

    async def count(self) -> int:
        """Вернуть число строк этой модели."""
        statement = select(func.count()).select_from(self.model)
        result = await self._session.execute(statement)
        return int(result.scalar_one())


class CharacterRepository(BaseRepository[Character], CharacterRepositoryProtocol):
    """Доступ на чтение к образам выживших, хранящимся в ``characters``."""

    model = Character

    async def get_all(self) -> Sequence[Character]:
        """Вернуть всех персонажей, упорядоченных по идентификатору."""
        statement = select(Character).order_by(Character.id)
        result = await self._session.execute(statement)
        return result.scalars().all()

    async def get_by_name(self, name: str) -> Character | None:
        """Найти персонажа по его уникальному отображаемому имени."""
        statement = select(Character).where(Character.name == name)
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()


class LoreChunkRepository(BaseRepository[LoreChunk], LoreChunkRepositoryProtocol):
    """Хранение и поиск по косинусной близости в базе знаний лора."""

    model = LoreChunk

    async def create_chunk(self, record: EmbeddedLoreChunk) -> LoreChunk:
        """Сохранить один проиндексированный фрагмент лора."""
        return await self.add(self._to_model(record))

    async def create_chunks(self, records: Sequence[EmbeddedLoreChunk]) -> Sequence[LoreChunk]:
        """Сохранить пакет проиндексированных фрагментов лора за один flush."""
        if not records:
            return []
        chunks = [self._to_model(record) for record in records]
        self._session.add_all(chunks)
        await self._session.flush()
        return chunks

    async def search_similar_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        source_types: Sequence[SourceType] | None = None,
        topic: str | None = None,
        character_name: str | None = None,
    ) -> Sequence[RetrievedChunk]:
        """Вернуть ``limit`` ближайших фрагментов по убыванию косинусной близости.

        pgvector выдаёт косинусное *расстояние* (``1 - similarity``); оператор ``<=>``
        строится через ``cosine_distance`` и может ускоряться HNSW-индексом,
        объявленным на модели.
        """
        distance = LoreChunk.embedding.cosine_distance(list(query_embedding)).label(
            "cosine_distance"
        )
        statement = select(LoreChunk, distance).order_by(distance).limit(limit)
        if source_types is not None:
            statement = statement.where(
                LoreChunk.source_type.in_([str(source_type) for source_type in source_types])
            )
        if topic is not None:
            statement = statement.where(LoreChunk.topic == topic)
        if character_name is not None:
            statement = statement.where(LoreChunk.character_name == character_name)

        result = await self._session.execute(statement)
        return [
            RetrievedChunk(chunk=chunk, similarity=1.0 - float(distance_value))
            for chunk, distance_value in result.all()
        ]

    async def delete_all(self) -> int:
        """Удалить все сохранённые фрагменты и сообщить, сколько строк ушло.

        Индексация инкрементальна — запись, чей текст изменился, сохраняется заново
        под новым хешем, а не заменяет старую строку, — поэтому повторная индексация
        корпуса, отредактированного на месте, сначала требует явной очистки. Метод
        намеренно отсутствует в порту репозитория: ни один сервис не вправе опустошить
        архив, только инструмент наполнения, владеющий его жизненным циклом.
        """
        result = await self._session.execute(delete(LoreChunk))
        # A DELETE returns a cursor result, which is the only kind that reports
        # how many rows it touched.
        assert isinstance(result, CursorResult)
        return int(result.rowcount or 0)

    async def existing_content_hashes(self, hashes: Sequence[str]) -> set[str]:
        """Вернуть, какие из данных хешей содержимого уже сохранены."""
        wanted = list(dict.fromkeys(hashes))
        found: set[str] = set()
        for start in range(0, len(wanted), HASH_LOOKUP_CHUNK_SIZE):
            window = wanted[start : start + HASH_LOOKUP_CHUNK_SIZE]
            statement = select(LoreChunk.content_hash).where(LoreChunk.content_hash.in_(window))
            result = await self._session.execute(statement)
            found.update(result.scalars().all())
        return found

    @staticmethod
    def _to_model(record: EmbeddedLoreChunk) -> LoreChunk:
        """Перевести запись корпуса в её ORM-представление."""
        return LoreChunk(
            content=record.content,
            content_hash=record.content_hash,
            source_type=str(record.source_type),
            topic=str(record.topic),
            title=record.title,
            character_name=record.character_name,
            source_url=record.source_url,
            embedding=list(record.embedding),
        )
