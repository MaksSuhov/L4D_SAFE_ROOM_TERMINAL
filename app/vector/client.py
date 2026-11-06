"""Владелец соединения с Qdrant вместе со схемой своих коллекций.

Единственный экземпляр создаётся в жизненном цикле приложения и живёт в
``app.state``, поэтому клиент и его пул соединений общие для процесса и не
опираются на глобальные переменные модуля. Схема коллекций принадлежит ему же:
только хранилище знает, из чего состоит его собственная коллекция, и это
позволяет наполнению не дублировать описание векторов.
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from app.config import Settings
from app.vector.schema import ensure_character_collection, ensure_lore_collection


class VectorStore:
    """Асинхронный клиент Qdrant вместе с именами своих коллекций."""

    def __init__(self, settings: Settings, *, client: AsyncQdrantClient | None = None) -> None:
        self._settings = settings
        # The client can be handed over ready, which is how a test or a local
        # in-process Qdrant shares one store between every part of the graph:
        # there is exactly one connection, and everyone sees the same collections.
        self._client = client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            prefer_grpc=settings.qdrant_prefer_grpc,
            timeout=settings.qdrant_timeout_seconds,
            check_compatibility=settings.qdrant_check_compatibility,
        )

    @property
    def client(self) -> AsyncQdrantClient:
        """Отдать клиент репозиториям и инструментам наполнения."""
        return self._client

    @property
    def lore_collection(self) -> str:
        """Имя коллекции архива лора."""
        return self._settings.lore_collection

    @property
    def character_collection(self) -> str:
        """Имя коллекции образов выживших."""
        return self._settings.character_collection

    async def ensure_collections(self, *, recreate: bool = False) -> None:
        """Привести обе коллекции к объявленной схеме.

        ``recreate`` — выключатель деструктивного инструмента: он опустошает
        архивы, чтобы изменившийся корпус заменял записи, а не наслаивался на них.
        """
        await ensure_lore_collection(
            self._client,
            self.lore_collection,
            dimensions=self._settings.embedding_dimensions,
            recreate=recreate,
        )
        await ensure_character_collection(
            self._client,
            self.character_collection,
            recreate=recreate,
        )

    async def ping(self) -> bool:
        """Проверить живость узла; используется пробой готовности."""
        try:
            await self._client.get_collections()
        except Exception:  # noqa: BLE001 - любой сбой транспорта значит «недоступен»
            return False
        return True

    async def close(self) -> None:
        """Освободить соединения; вызывается при остановке приложения."""
        await self._client.close()
