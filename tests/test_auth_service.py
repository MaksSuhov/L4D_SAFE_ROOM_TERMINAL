"""Проверки сервиса учётных записей: регистрация, вход, ротация и выход.

Стенд собран целиком на двойниках: настоящие часы, Argon2 и Qdrant здесь не
нужны, потому что проверяется не они, а порядок проверок доступа.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.domain import PasswordPolicy, User
from app.exceptions import (
    AccountInactiveError,
    AuthorizationError,
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
    InvalidTokenError,
    RefreshTokenReuseError,
    TokenExpiredError,
    WeakPasswordError,
)
from app.services import AuthService, JwtAccessTokenIssuer, SecureRefreshTokenFactory
from app.taxonomy import UserRole
from tests.fakes import (
    FakePasswordHasher,
    FixedClock,
    InMemoryRefreshSessionRepository,
    InMemoryUserRepository,
    build_user,
)

PASSWORD = "safe-room-passphrase"
ACCESS_TTL = 900
REFRESH_TTL = 3600


@dataclass
class AuthStand:
    """Сервис вместе с двойниками, которые ему подложены."""

    service: AuthService
    users: InMemoryUserRepository
    sessions: InMemoryRefreshSessionRepository
    hasher: FakePasswordHasher
    clock: FixedClock


def build_stand(
    *,
    users: list[User] | None = None,
    registration_enabled: bool = True,
) -> AuthStand:
    """Собрать сервис на двойниках с предсказуемыми часами."""
    clock = FixedClock()
    hasher = FakePasswordHasher()
    user_repository = InMemoryUserRepository(users)
    session_repository = InMemoryRefreshSessionRepository()
    service = AuthService(
        user_repository,
        session_repository,
        hasher,
        JwtAccessTokenIssuer(
            "s" * 48,
            algorithm="HS256",
            issuer="l4d",
            audience="terminal",
            ttl_seconds=ACCESS_TTL,
            clock=clock,
        ),
        SecureRefreshTokenFactory(),
        clock,
        policy=PasswordPolicy(),
        access_ttl_seconds=ACCESS_TTL,
        refresh_ttl_seconds=REFRESH_TTL,
        registration_enabled=registration_enabled,
    )
    return AuthStand(service, user_repository, session_repository, hasher, clock)


async def test_registration_opens_a_member_session() -> None:
    stand = build_stand()

    issued = await stand.service.register(email="Ellis@L4D.dev", password=PASSWORD)

    assert issued.user.email == "ellis@l4d.dev"
    assert issued.user.role is UserRole.MEMBER
    assert issued.user.is_admin is False
    assert issued.user.display_name == "ellis"
    assert issued.expires_in == ACCESS_TTL
    assert issued.refresh_expires_in == REFRESH_TTL
    assert stand.hasher.hashed == [PASSWORD]
    assert len(stand.sessions.sessions) == 1


async def test_registration_keeps_the_requested_display_name() -> None:
    stand = build_stand()

    issued = await stand.service.register(
        email="nick@l4d.dev",
        password=PASSWORD,
        display_name="  Ник  ",
    )

    assert issued.user.display_name == "Ник"


async def test_a_taken_email_is_refused() -> None:
    stand = build_stand(users=[build_user(email="ellis@l4d.dev")])

    with pytest.raises(EmailAlreadyRegisteredError):
        await stand.service.register(email="ELLIS@l4d.dev", password=PASSWORD)

    assert stand.hasher.hashed == []


@pytest.mark.parametrize(
    "password",
    ["short", "a" * 129, " qwertyuiop", "password123"],
)
async def test_a_weak_password_is_refused(password: str) -> None:
    """Пароль проверяется до всего остального: учётной записи для него ещё нет."""
    stand = build_stand()

    with pytest.raises(WeakPasswordError):
        await stand.service.register(email="ellis@l4d.dev", password=password)

    assert await stand.users.count() == 0


async def test_registration_can_be_closed_by_configuration() -> None:
    stand = build_stand(registration_enabled=False)

    with pytest.raises(AuthorizationError):
        await stand.service.register(email="ellis@l4d.dev", password=PASSWORD)


async def test_provisioning_creates_an_operator_without_opening_a_session() -> None:
    """Первый оператор заводится по долгу службы, а не самостоятельной заявкой."""
    stand = build_stand()

    user = await stand.service.provision(
        email="operator@l4d.dev",
        password=PASSWORD,
        display_name="Оператор",
        role=UserRole.ADMIN,
    )

    assert user.is_admin is True
    assert user.role is UserRole.ADMIN
    assert stand.sessions.sessions == []


async def test_provisioning_works_while_registration_is_closed() -> None:
    """Выключенная регистрация закрывает дверь всем, кроме того, кто заводит записи."""
    stand = build_stand(registration_enabled=False)

    user = await stand.service.provision(email="operator@l4d.dev", password=PASSWORD)

    assert user.role is UserRole.MEMBER
    assert await stand.users.count() == 1


async def test_login_ignores_the_case_of_the_email() -> None:
    stand = build_stand(users=[build_user(password_hash=f"fake${PASSWORD}")])

    issued = await stand.service.log_in(email="  ELLIS@l4d.dev ", password=PASSWORD)

    assert issued.user.email == "ellis@l4d.dev"
    assert stand.hasher.verified == [PASSWORD]


async def test_an_unknown_email_costs_the_same_time_as_a_wrong_password() -> None:
    """Иначе по скорости ответа видно, какие адреса на терминале заведены."""
    stand = build_stand(users=[build_user(password_hash=f"fake${PASSWORD}")])

    with pytest.raises(InvalidCredentialsError):
        await stand.service.log_in(email="nobody@l4d.dev", password=PASSWORD)

    assert stand.hasher.verified == []
    assert stand.hasher.equalised == [PASSWORD]


async def test_a_wrong_password_is_refused() -> None:
    stand = build_stand(users=[build_user(password_hash=f"fake${PASSWORD}")])

    with pytest.raises(InvalidCredentialsError):
        await stand.service.log_in(email="ellis@l4d.dev", password="another-passphrase")

    assert stand.sessions.sessions == []


async def test_a_blocked_account_cannot_log_in() -> None:
    stand = build_stand(users=[build_user(password_hash=f"fake${PASSWORD}", is_active=False)])

    with pytest.raises(AccountInactiveError):
        await stand.service.log_in(email="ellis@l4d.dev", password=PASSWORD)


async def test_a_valid_password_rehashes_an_outdated_hash() -> None:
    """Параметры алгоритма меняются, и старый хеш обновляется при первом же входе."""
    stand = build_stand(users=[build_user(password_hash=f"fake${PASSWORD}")])
    stand.hasher.stale.add(f"fake${PASSWORD}")

    await stand.service.log_in(email="ellis@l4d.dev", password=PASSWORD)

    assert stand.hasher.hashed == [PASSWORD]
    assert stand.hasher.verified == [PASSWORD]


async def test_refresh_rotates_the_pair_and_retires_the_presented_token() -> None:
    stand = build_stand()
    first = await stand.service.register(email="ellis@l4d.dev", password=PASSWORD)

    second = await stand.service.refresh(first.refresh_token)

    assert second.refresh_token != first.refresh_token
    assert second.access_token.value != first.access_token.value
    assert len(stand.sessions.sessions) == 2
    assert stand.sessions.sessions[0].revoked_at == stand.clock.now()
    assert stand.sessions.sessions[0].replaced_by == stand.sessions.sessions[1].id
    assert stand.sessions.sessions[0].family_id == stand.sessions.sessions[1].family_id


async def test_reusing_a_rotated_token_retires_the_whole_family() -> None:
    """Повторное предъявление — признак кражи, и разбираться, кто есть кто, поздно."""
    stand = build_stand()
    first = await stand.service.register(email="ellis@l4d.dev", password=PASSWORD)
    second = await stand.service.refresh(first.refresh_token)

    with pytest.raises(RefreshTokenReuseError):
        await stand.service.refresh(first.refresh_token)

    assert stand.sessions.active == []
    with pytest.raises(RefreshTokenReuseError):
        await stand.service.refresh(second.refresh_token)


async def test_an_expired_session_is_retired_and_refused() -> None:
    stand = build_stand()
    issued = await stand.service.register(email="ellis@l4d.dev", password=PASSWORD)

    stand.clock.advance(REFRESH_TTL + 1)

    with pytest.raises(TokenExpiredError):
        await stand.service.refresh(issued.refresh_token)

    assert stand.sessions.sessions[0].is_revoked


async def test_an_unknown_refresh_token_is_refused() -> None:
    stand = build_stand()

    with pytest.raises(InvalidTokenError):
        await stand.service.refresh("a-token-nobody-issued")


async def test_refresh_of_a_blocked_account_retires_its_sessions() -> None:
    """Блокировка должна закрывать и путь обновления, а не только вход."""
    stand = build_stand()
    issued = await stand.service.register(email="ellis@l4d.dev", password=PASSWORD)
    user = await stand.users.get_by_email("ellis@l4d.dev")
    assert user is not None
    user.is_active = False

    with pytest.raises(AccountInactiveError):
        await stand.service.refresh(issued.refresh_token)

    assert stand.sessions.active == []


async def test_the_access_token_resolves_to_the_record() -> None:
    stand = build_stand()
    issued = await stand.service.register(email="ellis@l4d.dev", password=PASSWORD)

    resolved = await stand.service.resolve_access_token(issued.access_token.value)

    assert resolved.email == "ellis@l4d.dev"


async def test_rights_come_from_the_record_and_not_from_the_token() -> None:
    """Понижение и повышение роли действуют сразу, а не вместе со сроком токена."""
    stand = build_stand()
    issued = await stand.service.register(email="ellis@l4d.dev", password=PASSWORD)
    user = await stand.users.get_by_email("ellis@l4d.dev")
    assert user is not None

    user.role = UserRole.ADMIN

    resolved = await stand.service.resolve_access_token(issued.access_token.value)
    assert resolved.is_admin is True


async def test_the_access_token_of_a_deleted_account_is_refused() -> None:
    stand = build_stand()
    issued = await stand.service.register(email="ellis@l4d.dev", password=PASSWORD)
    await stand.users.delete(issued.user.identifier)

    with pytest.raises(InvalidTokenError):
        await stand.service.resolve_access_token(issued.access_token.value)


async def test_log_out_retires_only_the_presented_session() -> None:
    stand = build_stand()
    laptop = await stand.service.register(email="ellis@l4d.dev", password=PASSWORD)
    desktop = await stand.service.log_in(email="ellis@l4d.dev", password=PASSWORD)

    await stand.service.log_out(laptop.refresh_token)

    assert len(stand.sessions.active) == 1
    refreshed = await stand.service.refresh(desktop.refresh_token)
    assert refreshed.user.email == "ellis@l4d.dev"


async def test_log_out_is_idempotent_and_quiet_without_a_token() -> None:
    stand = build_stand()
    issued = await stand.service.register(email="ellis@l4d.dev", password=PASSWORD)

    await stand.service.log_out(issued.refresh_token)
    await stand.service.log_out(issued.refresh_token)
    await stand.service.log_out(None)

    assert stand.sessions.active == []


async def test_log_out_everywhere_counts_the_sessions_it_retires() -> None:
    stand = build_stand()
    await stand.service.register(email="ellis@l4d.dev", password=PASSWORD)
    await stand.service.log_in(email="ellis@l4d.dev", password=PASSWORD)

    revoked = await stand.service.log_out_everywhere("user-1")

    assert revoked == 2
    assert await stand.service.log_out_everywhere("user-1") == 0
