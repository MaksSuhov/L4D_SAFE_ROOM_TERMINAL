"""Database layer: declarative base, async engine and session factory.

The module owns *how* connections are created and never contains SQL for a
specific entity — that responsibility belongs to the repository layer.
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
    """Declarative base shared by every ORM model of the project."""


class Database:
    """Owns the async engine and hands out short-lived sessions.

    A single instance is created during the FastAPI lifespan and stored in
    ``app.state``, so the connection pool is shared by the whole process
    without relying on module-level globals.
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
        """Expose the engine for health checks and tests."""
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Provide a session that commits on success and rolls back on failure."""
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
        """Enable pgvector and create every table declared on :class:`Base`."""
        # Imported here to guarantee that all mappers are registered in the
        # metadata before ``create_all`` runs, regardless of import order.
        import app.models  # noqa: F401  pylint: disable=unused-import

        async with self._engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.run_sync(Base.metadata.create_all)

    async def drop_all(self) -> None:
        """Drop every table of the project; used by destructive tooling."""
        import app.models  # noqa: F401  pylint: disable=unused-import

        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)

    async def dispose(self) -> None:
        """Close the connection pool; called on application shutdown."""
        await self._engine.dispose()
