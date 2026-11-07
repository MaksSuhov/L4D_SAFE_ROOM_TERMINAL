from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Final

from qdrant_client import AsyncQdrantClient, models

from app.domain import User, normalise_email
from app.exceptions import EmailAlreadyRegisteredError, ResourceNotFoundError
from app.taxonomy import UserRole

USER_POINT_NAMESPACE: Final[uuid.UUID] = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")
"""Пространство имён, из которого выводится идентификатор учётной записи."""

SCROLL_PAGE_SIZE: Final[int] = 128
"""Размер страницы при чтении коллекции: записей мало, но пагинация делает
чтение верным для любого их числа.
"""


class QdrantUserRepository:
    """Хранение и чтение учётных записей.

    Идентификатор записи выводится из адреса почты, а не выдаётся случайно: у
    Qdrant нет ограничения уникальности, и такая схема гарантирует, что на один
    адрес всегда приходится ровно одна учётная запись. Проверка занятости перед
    записью не атомарна: две одновременные регистрации на один адрес разойдутся
    тем, что вторая перезапишет первую, а не заведёт дубль, — потерять гонку
    лучше, чем получить два разных пароля под одним логином.
    """

    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    async def get_by_id(self, entity_id: str) -> User | None:
        """Вернуть учётную запись с данным идентификатором или ``None``."""
        records = await self._retrieve(entity_id)
        return self._to_user(records[0]) if records else None

    async def get_by_email(self, email: str) -> User | None:
        """Найти учётную запись по адресу почты."""
        records = await self._scroll(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="email",
                        match=models.MatchValue(value=normalise_email(email)),
                    )
                ]
            )
        )
        return self._to_user(records[0]) if records else None

    async def get_all(self) -> Sequence[User]:
        """Вернуть все учётные записи, упорядоченные по адресу почты."""
        records = await self._scroll()
        users = [self._to_user(record) for record in records]
        return sorted(users, key=lambda user: user.email)

    async def create(self, user: User) -> None:
        """Сохранить новую запись, выдав ей идентификатор."""
        if await self.get_by_email(user.email) is not None:
            raise EmailAlreadyRegisteredError(user.email)

        identifier = str(uuid.uuid5(USER_POINT_NAMESPACE, normalise_email(user.email)))
        await self._client.upsert(
            self._collection,
            points=[models.PointStruct(id=identifier, vector={}, payload=self._to_payload(user))],
            wait=True,
        )
        user.id = identifier

    async def update(self, user: User) -> None:
        """Записать изменения существующей учётной записи."""
        if await self.get_by_id(user.identifier) is None:
            raise ResourceNotFoundError("User", user.identifier)

        await self._client.upsert(
            self._collection,
            points=[
                models.PointStruct(
                    id=user.identifier,
                    vector={},
                    payload=self._to_payload(user),
                )
            ],
            wait=True,
        )

    async def delete(self, entity_id: str) -> bool:
        """Удалить учётную запись и сообщить, была ли она вообще."""
        if not await self._retrieve(entity_id):
            return False
        await self._client.delete(
            self._collection,
            points_selector=models.PointIdsList(points=[entity_id]),
            wait=True,
        )
        return True

    async def count(self) -> int:
        """Вернуть число учётных записей."""
        response = await self._client.count(collection_name=self._collection, exact=True)
        return response.count

    async def _retrieve(self, entity_id: str) -> list[models.Record]:
        """Прочитать точку по идентификатору, если она есть."""
        records = await self._client.retrieve(
            collection_name=self._collection,
            ids=[entity_id],
            with_payload=True,
            with_vectors=False,
        )
        return list(records)

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
    def _to_payload(user: User) -> dict[str, object]:
        """Разложить учётную запись по полям payload."""
        return {
            "email": user.email,
            "display_name": user.display_name,
            "password_hash": user.password_hash,
            "role": str(user.role),
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }

    @staticmethod
    def _to_user(record: models.Record) -> User:
        """Собрать доменную сущность из точки.

        Идентификатор берётся у самой точки: он же служит её ключом, поэтому
        дублировать его в payload не нужно — иначе два источника правды однажды
        разошлись бы.
        """
        payload = record.payload or {}
        created_at = payload.get("created_at")
        return User(
            email=str(payload["email"]),
            display_name=str(payload.get("display_name", "")),
            password_hash=str(payload["password_hash"]),
            role=UserRole(str(payload.get("role", UserRole.MEMBER))),
            is_active=bool(payload.get("is_active", True)),
            created_at=datetime.fromisoformat(str(created_at)) if created_at else None,
            id=str(record.id),
        )
