"""Сквозные тесты учётных записей против Qdrant в локальном режиме.

Двойники из ``tests/fakes.py`` доказывают логику доступа, но не проверяют сами
запросы: фильтр по непогашенным сессиям, частичное обновление payload, удаление
точки. Эти вызовы проверяются здесь на настоящем Qdrant, с настоящим Argon2 —
сервер, который принимает такие фильтры, единственный авторитет по ним.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
from qdrant_client import AsyncQdrantClient

from app.config import Settings, get_settings
from app.dependencies import (
    build_access_token_issuer,
    build_auth_service,
    build_password_hasher,
    build_refresh_token_factory,
)
from app.exceptions import ApplicationError, EmailAlreadyRegisteredError, RefreshTokenReuseError
from app.services import AuthService, SecureRefreshTokenFactory, SystemClock
from app.taxonomy import UserRole
from app.vector import QdrantRefreshSessionRepository, QdrantUserRepository, VectorStore

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

PASSWORD = "safe-room-passphrase"
EMAIL = "operator@l4d.dev"


@dataclass
class AuthTerminal:
    """Сервис учётных записей поверх настоящего локального Qdrant."""

    settings: Settings
    service: AuthService
    users: QdrantUserRepository
    sessions: QdrantRefreshSessionRepository
    client: AsyncQdrantClient


@pytest.fixture
async def terminal() -> AsyncIterator[AuthTerminal]:
    """Поднять хранилище в процессе и собрать на нём сервис учётных записей."""
    settings = get_settings()
    client = AsyncQdrantClient(":memory:")
    store = VectorStore(settings, client=client)
    await store.ensure_collections()

    users = QdrantUserRepository(client, store.user_collection)
    sessions = QdrantRefreshSessionRepository(client, store.refresh_session_collection)
    clock = SystemClock()
    service = build_auth_service(
        users,
        sessions,
        build_password_hasher(),
        build_access_token_issuer(settings, clock),
        build_refresh_token_factory(),
        clock,
        settings,
    )

    try:
        yield AuthTerminal(settings, service, users, sessions, client)
    finally:
        await client.close()


async def test_an_account_survives_a_round_trip_through_qdrant(terminal: AuthTerminal) -> None:
    user = await terminal.service.provision(
        email="Operator@L4D.dev",
        password=PASSWORD,
        display_name="Оператор",
        role=UserRole.ADMIN,
    )

    stored = await terminal.users.get_by_email("OPERATOR@l4d.dev")

    assert stored is not None
    assert stored.id == user.id
    assert stored.email == EMAIL
    assert stored.role is UserRole.ADMIN
    assert stored.display_name == "Оператор"
    assert stored.is_active is True
    assert stored.created_at is not None
    assert await terminal.users.count() == 1
    # Хеш настоящий: Argon2 проверяется, а сам пароль в хранилище не лежит.
    assert stored.password_hash.startswith("$argon2")
    assert PASSWORD not in stored.password_hash


async def test_the_stored_password_verifies_with_the_real_hasher(terminal: AuthTerminal) -> None:
    await terminal.service.provision(email=EMAIL, password=PASSWORD)

    issued = await terminal.service.log_in(email=EMAIL, password=PASSWORD)

    assert issued.user.email == EMAIL
    assert issued.access_token.value


async def test_according_to_the_store_a_second_registration_is_a_duplicate(
    terminal: AuthTerminal,
) -> None:
    await terminal.service.provision(email=EMAIL, password=PASSWORD)

    with pytest.raises(EmailAlreadyRegisteredError):
        await terminal.service.provision(email=EMAIL, password=PASSWORD)


async def test_a_stored_account_can_be_updated_in_place(terminal: AuthTerminal) -> None:
    user = await terminal.service.provision(email=EMAIL, password=PASSWORD)

    user.role = UserRole.ADMIN
    user.is_active = False
    await terminal.users.update(user)

    stored = await terminal.users.get_by_id(user.identifier)
    assert stored is not None
    assert stored.role is UserRole.ADMIN
    assert stored.is_active is False
    assert await terminal.users.count() == 1


async def test_updating_an_account_that_is_gone_is_refused(terminal: AuthTerminal) -> None:
    user = await terminal.service.provision(email=EMAIL, password=PASSWORD)
    await terminal.users.delete(user.identifier)

    with pytest.raises(ApplicationError):
        await terminal.users.update(user)


async def test_a_deleted_account_leaves_the_store(terminal: AuthTerminal) -> None:
    user = await terminal.service.provision(email=EMAIL, password=PASSWORD)

    assert await terminal.users.delete(user.identifier) is True
    assert await terminal.users.delete(user.identifier) is False
    assert await terminal.users.get_by_email(EMAIL) is None


async def test_rotation_retires_the_presented_token_in_the_store(
    terminal: AuthTerminal,
) -> None:
    factory = SecureRefreshTokenFactory()
    first = await terminal.service.provision(email=EMAIL, password=PASSWORD)
    session = await terminal.service.log_in(email=EMAIL, password=PASSWORD)
    assert first.email == session.user.email

    rotated = await terminal.service.refresh(session.refresh_token)

    retired = await terminal.sessions.get_by_digest(factory.digest(session.refresh_token))
    successor = await terminal.sessions.get_by_digest(factory.digest(rotated.refresh_token))
    assert retired is not None and retired.revoked_at is not None
    assert successor is not None and successor.revoked_at is None
    assert retired.replaced_by == successor.identifier
    assert retired.family_id == successor.family_id


async def test_replaying_a_token_retires_the_chain_in_the_store(
    terminal: AuthTerminal,
) -> None:
    factory = SecureRefreshTokenFactory()
    issued = await terminal.service.register(email=EMAIL, password=PASSWORD)
    rotated = await terminal.service.refresh(issued.refresh_token)

    with pytest.raises(RefreshTokenReuseError):
        await terminal.service.refresh(issued.refresh_token)

    successor = await terminal.sessions.get_by_digest(factory.digest(rotated.refresh_token))
    assert successor is not None
    assert successor.revoked_at is not None


async def test_log_out_everywhere_retires_every_stored_session(
    terminal: AuthTerminal,
) -> None:
    factory = SecureRefreshTokenFactory()
    first = await terminal.service.register(email=EMAIL, password=PASSWORD)
    second = await terminal.service.log_in(email=EMAIL, password=PASSWORD)

    revoked = await terminal.service.log_out_everywhere(first.user.identifier)

    assert revoked == 2
    for token in (first.refresh_token, second.refresh_token):
        stored = await terminal.sessions.get_by_digest(factory.digest(token))
        assert stored is not None and stored.revoked_at is not None
    # Второй раз гасить нечего, и это не ошибка.
    assert await terminal.service.log_out_everywhere(first.user.identifier) == 0


async def test_log_out_retires_only_the_named_session(terminal: AuthTerminal) -> None:
    factory = SecureRefreshTokenFactory()
    laptop = await terminal.service.register(email=EMAIL, password=PASSWORD)
    desktop = await terminal.service.log_in(email=EMAIL, password=PASSWORD)

    await terminal.service.log_out(laptop.refresh_token)

    retired = await terminal.sessions.get_by_digest(factory.digest(laptop.refresh_token))
    alive = await terminal.sessions.get_by_digest(factory.digest(desktop.refresh_token))
    assert retired is not None and retired.revoked_at is not None
    assert alive is not None and alive.revoked_at is None
