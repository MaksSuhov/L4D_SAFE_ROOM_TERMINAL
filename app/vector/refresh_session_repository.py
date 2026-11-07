from __future__ import annotations

import uuid
from datetime import datetime
from typing import Final

from qdrant_client import AsyncQdrantClient, models

from app.domain import RefreshSession

SCROLL_PAGE_SIZE: Final[int] = 128
"""Размер страницы при чтении коллекции сессий."""

REVOKED_AT: Final[str] = "revoked_at"
"""Поле, отличающее погашенную сессию от действующей: пустое значение — действующая."""


class QdrantRefreshSessionRepository:
    """Хранение, поиск и отзыв сессий, выданных по refresh-токенам.

    Записи не удаляются, а гасятся: цепочка ротаций — это след, по которому видно
    и повторное предъявление токена, и то, что сессия закончилась не сама. Ради
    этого коллекция растёт медленнее, чем число входов, но не бесконтрольно:
    погашенная сессия занимает одну строку на всю свою жизнь.
    """

    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    async def create(self, session: RefreshSession) -> None:
        """Сохранить сессию, выдав ей идентификатор."""
        identifier = str(uuid.uuid4())
        await self._client.upsert(
            self._collection,
            points=[
                models.PointStruct(
                    id=identifier,
                    vector={},
                    payload={
                        "user_id": session.user_id,
                        "token_digest": session.token_digest,
                        "family_id": session.family_id,
                        "issued_at": session.issued_at.isoformat(),
                        "expires_at": session.expires_at.isoformat(),
                        REVOKED_AT: None,
                        "replaced_by": None,
                        "user_agent": session.user_agent,
                    },
                )
            ],
            wait=True,
        )
        session.id = identifier

    async def get_by_digest(self, digest: str) -> RefreshSession | None:
        """Найти сессию по отпечатку предъявленного токена."""
        records = await self._scroll(digest_filter(digest), limit=1)
        return self._to_session(records[0]) if records else None

    async def revoke(
        self,
        session_id: str,
        *,
        revoked_at: datetime,
        replaced_by: str | None = None,
    ) -> None:
        """Погасить одну сессию, оставив след о том, кто её заменил."""
        await self._client.set_payload(
            collection_name=self._collection,
            payload={REVOKED_AT: revoked_at.isoformat(), "replaced_by": replaced_by},
            points=[session_id],
            wait=True,
        )

    async def revoke_family(self, family_id: str, *, revoked_at: datetime) -> int:
        """Погасить всю цепочку ротаций и сообщить, сколько сессий было отозвано."""
        active = await self._scroll(active_filter("family_id", family_id))
        return await self._revoke_many([str(record.id) for record in active], revoked_at)

    async def revoke_all_for_user(self, user_id: str, *, revoked_at: datetime) -> int:
        """Погасить все сессии учётной записи и сообщить, сколько их было."""
        active = await self._scroll(active_filter("user_id", user_id))
        return await self._revoke_many([str(record.id) for record in active], revoked_at)

    async def _revoke_many(self, identifiers: list[str], revoked_at: datetime) -> int:
        """Погасить перечисленные сессии одним запросом."""
        if not identifiers:
            return 0
        points: list[models.ExtendedPointId] = list(identifiers)
        await self._client.set_payload(
            collection_name=self._collection,
            payload={REVOKED_AT: revoked_at.isoformat()},
            points=points,
            wait=True,
        )
        return len(identifiers)

    async def _scroll(
        self,
        scroll_filter: models.Filter,
        *,
        limit: int = SCROLL_PAGE_SIZE,
    ) -> list[models.Record]:
        """Прочитать подходящие записи страницами."""
        records: list[models.Record] = []
        offset: models.ExtendedPointId | None = None
        while True:
            page, offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=scroll_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(page)
            if offset is None or len(records) >= limit:
                return records

    @staticmethod
    def _to_session(record: models.Record) -> RefreshSession:
        """Собрать доменную сущность из точки."""
        payload = record.payload or {}
        revoked_at = payload.get(REVOKED_AT)
        replaced_by = payload.get("replaced_by")
        return RefreshSession(
            user_id=str(payload["user_id"]),
            token_digest=str(payload["token_digest"]),
            family_id=str(payload["family_id"]),
            issued_at=datetime.fromisoformat(str(payload["issued_at"])),
            expires_at=datetime.fromisoformat(str(payload["expires_at"])),
            revoked_at=datetime.fromisoformat(str(revoked_at)) if revoked_at else None,
            replaced_by=str(replaced_by) if replaced_by else None,
            user_agent=str(payload.get("user_agent", "")),
            id=str(record.id),
        )


def digest_filter(digest: str) -> models.Filter:
    """Отобрать сессию по отпечатку токена."""
    return models.Filter(
        must=[
            models.FieldCondition(key="token_digest", match=models.MatchValue(value=digest)),
        ]
    )


def active_filter(field: str, value: str) -> models.Filter:
    """Отобрать действующие сессии по владельцу или цепочке ротаций."""
    return models.Filter(
        must=[
            models.FieldCondition(key=field, match=models.MatchValue(value=value)),
            models.IsNullCondition(is_null=models.PayloadField(key=REVOKED_AT)),
        ]
    )
