"""Слой базы данных: декларативная база, асинхронный движок и фабрика сессий.

Модуль отвечает за то, *как* создаются соединения, и никогда не содержит SQL
для конкретной сущности — это ответственность слоя репозиториев.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import Settings


class Base(DeclarativeBase):
    """Декларативная база, общая для всех ORM-моделей проекта."""


class Database:
    """Владеет асинхронным движком и выдаёт недолговечные сессии.

    Единственный экземпляр создаётся на протяжении жизненного цикла FastAPI и
    хранится в ``app.state``, поэтому пул соединений общий для всего процесса и не
    опирается на глобальные переменные модуля.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine: AsyncEngine = create_async_engine(
            settings.database_url,
            echo=settings.database_echo,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_pre_ping=True,
        )
        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        """Отдать движок для проверки живости и тестов."""
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Отдать сессию, которая коммитит при успехе и откатывается при ошибке."""
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def create_all(self) -> None:
        """Включить pgvector и создать все таблицы, объявленные на :class:`Base`."""
        # Imported here to guarantee that all mappers are registered in the
        # metadata before ``create_all`` runs, regardless of import order.
        import app.models  # noqa: F401  pylint: disable=unused-import

        async with self._engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.run_sync(Base.metadata.create_all)

    async def drop_all(self) -> None:
        """Удалить все таблицы проекта; используется деструктивным инструментом."""
        import app.models  # noqa: F401  pylint: disable=unused-import

        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)

    async def dispose(self) -> None:
        """Закрыть пул соединений; вызывается при остановке приложения."""
        await self._engine.dispose()
