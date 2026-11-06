"""Репозиторий образов выживших на Qdrant.

Образы не векторизуются: их никто не ищет по близости, и коллекция для них —
просто хранилище с фильтрами по имени и идентификатору. Реляционная база ради
восьми строк промптов стоила бы дороже, чем стоит вся остальная инфраструктура
проекта, поэтому она здесь и не нужна.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient, models

from app.domain import Character

SCROLL_PAGE_SIZE: int = 128
"""Размер страницы при чтении коллекции: образов мало, но пагинация делает
чтение верным для любого их числа.
"""


class QdrantCharacterRepository:
    """Хранение и чтение образов выживших."""

    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    async def get_by_id(self, entity_id: int) -> Character | None:
        """Вернуть образ с данным идентификатором или ``None``."""
        records = await self._scroll(
            models.Filter(
                must=[models.FieldCondition(key="id", match=models.MatchValue(value=entity_id))]
            )
        )
        return self._to_character(records[0].payload or {}) if records else None

    async def get_by_name(self, name: str) -> Character | None:
        """Найти образ по его уникальному отображаемому имени."""
        records = await self._scroll(
            models.Filter(
                must=[models.FieldCondition(key="name", match=models.MatchValue(value=name))]
            )
        )
        return self._to_character(records[0].payload or {}) if records else None

    async def get_all(self) -> Sequence[Character]:
        """Вернуть всех сохранённых персонажей, упорядоченных по идентификатору."""
        records = await self._scroll()
        characters = [self._to_character(record.payload or {}) for record in records]
        return sorted(characters, key=lambda character: character.id or 0)

    async def save(self, character: Character) -> bool:
        """Записать образ, обновив его на месте, если он уже есть.

        Возвращает признак того, что образ добавлен впервые: инструменту
        наполнения это нужно только для отчёта о прогоне, и он не должен ради него
        читать коллекцию дважды.
        """
        existing = await self.get_by_name(character.name)
        identifier = existing.id if existing is not None else await self._next_id()

        await self._client.upsert(
            self._collection,
            points=[
                models.PointStruct(
                    id=self.point_id(character.name),
                    vector={},
                    payload={
                        "id": identifier,
                        "name": character.name,
                        "system_prompt": character.system_prompt,
                        "few_shot_examples": list(character.few_shot_examples),
                    },
                )
            ],
            wait=True,
        )
        return existing is None

    @staticmethod
    def point_id(name: str) -> int:
        """Вывести идентификатор точки из имени образа.

        Имя уникально, поэтому служит ключом записи: повторное сохранение
        перезаписывает тот же образ, а не заводит второй.
        """
        digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).hexdigest()
        return int(digest, 16)

    async def _next_id(self) -> int:
        """Выдать следующий свободный идентификатор образа."""
        records = await self._scroll()
        identifiers = [int(str(record.payload["id"])) for record in records if record.payload]
        return max(identifiers, default=0) + 1

    async def _scroll(self, scroll_filter: models.Filter | None = None) -> list[models.Record]:
        """Прочитать коллекцию страницами, пока сервер не скажет, что она кончилась."""
        records: list[models.Record] = []
        offset: models.ExtendedPointId | None = None
        while True:
            page, offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=scroll_filter,
                limit=SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(page)
            if offset is None:
                return records

    @staticmethod
    def _to_character(payload: dict[str, object]) -> Character:
        """Собрать доменную сущность из payload точки.

        Значения payload приходят без типа, поэтому идентификатор приводится через
        строку: так одно и то же выражение принимает и число, и его текстовую форму.
        """
        examples = payload.get("few_shot_examples")
        return Character(
            name=str(payload["name"]),
            system_prompt=str(payload["system_prompt"]),
            few_shot_examples=(
                [dict(example) for example in examples] if isinstance(examples, list) else []
            ),
            id=int(str(payload["id"])),
        )
