from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

from app.domain import PasswordPolicy, RefreshSession, User, normalise_email
from app.exceptions import (
    AccountInactiveError,
    AuthorizationError,
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
    InvalidTokenError,
    RefreshTokenReuseError,
    TokenExpiredError,
)
from app.interfaces import (
    AccessTokenIssuerProtocol,
    ClockProtocol,
    IssuedAccessToken,
    PasswordHasherProtocol,
    RefreshSessionRepositoryProtocol,
    RefreshTokenFactoryProtocol,
    UserRepositoryProtocol,
)
from app.taxonomy import UserRole

CSRF_TOKEN_BYTES: Final[int] = 32
"""Энтропия CSRF-токена: он защищает от подстановки запроса, а не от перебора."""


@dataclass(frozen=True, slots=True)
class IssuedSession:
    """Итог установления сессии: то, что слой API раскладывает по cookie."""

    user: User
    access_token: IssuedAccessToken
    refresh_token: str
    refresh_expires_at: datetime
    csrf_token: str
    expires_in: int
    """Срок жизни токена доступа в секундах: его клиент показывает, не считая сам."""
    refresh_expires_in: int
    """Срок жизни refresh-токена в секундах: это же значение ставится в cookie."""


class AuthService:
    """Учётные записи и сессии: регистрация, вход, ротация токенов и выход.

    Сервис владеет всем жизненным циклом пары «доступ — обновление» и единственный
    знает, когда сессия считается скомпрометированной. Хранилище, часы и алгоритмы
    приходят портами, поэтому ни одна из этих проверок не зависит от Qdrant,
    от системного времени или от конкретной библиотеки подписи.
    """

    def __init__(
        self,
        users: UserRepositoryProtocol,
        sessions: RefreshSessionRepositoryProtocol,
        hasher: PasswordHasherProtocol,
        access_tokens: AccessTokenIssuerProtocol,
        refresh_tokens: RefreshTokenFactoryProtocol,
        clock: ClockProtocol,
        *,
        policy: PasswordPolicy,
        access_ttl_seconds: int,
        refresh_ttl_seconds: int,
        registration_enabled: bool = True,
    ) -> None:
        self._users = users
        self._sessions = sessions
        self._hasher = hasher
        self._access_tokens = access_tokens
        self._refresh_tokens = refresh_tokens
        self._clock = clock
        self._policy = policy
        self._access_ttl_seconds = access_ttl_seconds
        self._refresh_ttl_seconds = refresh_ttl_seconds
        self._registration_enabled = registration_enabled

    async def register(
        self,
        *,
        email: str,
        password: str,
        display_name: str | None = None,
        user_agent: str = "",
    ) -> IssuedSession:
        """Завести учётную запись выжившего и сразу открыть сессию."""
        if not self._registration_enabled:
            raise AuthorizationError(
                "Самостоятельная регистрация выключена: учётные записи заводит администратор."
            )

        user = await self.provision(
            email=email,
            password=password,
            display_name=display_name,
        )
        issued, _ = await self._open_session(user, user_agent=user_agent)
        return issued

    async def provision(
        self,
        *,
        email: str,
        password: str,
        display_name: str | None = None,
        role: UserRole = UserRole.MEMBER,
    ) -> User:
        """Завести учётную запись с заданной ролью, не открывая сессию.

        Этим путём заводит записи тот, у кого это служебная обязанность: первый
        оператор нужен раньше, чем появится кому регистрировать остальных, а
        выключенная самостоятельная регистрация не должна мешать заводить людей
        вручную.
        """
        normalised = normalise_email(email)
        self._policy.validate(password)
        if await self._users.get_by_email(normalised) is not None:
            raise EmailAlreadyRegisteredError(normalised)

        # Подпись из одних пробелов — это отсутствие подписи; так же считает и
        # схема запроса, поэтому сервис не заводит второго толкования.
        requested_name = (display_name or "").strip()
        user = User(
            email=normalised,
            password_hash=self._hasher.hash(password),
            display_name=requested_name or _display_name_from_email(normalised),
            role=role,
            created_at=self._clock.now(),
        )
        await self._users.create(user)
        return user

    async def log_in(self, *, email: str, password: str, user_agent: str = "") -> IssuedSession:
        """Проверить пару «почта — пароль» и открыть сессию."""
        user = await self._users.get_by_email(normalise_email(email))
        if user is None:
            # Проверять нечего, но время на ответ должно уйти то же самое: иначе
            # по скорости ответа видно, какие адреса на терминале заведены.
            self._hasher.equalise_timing(password)
            raise InvalidCredentialsError()
        if not self._hasher.verify(password, user.password_hash):
            raise InvalidCredentialsError()
        if not user.is_active:
            raise AccountInactiveError()

        await self._upgrade_hash_if_stale(user, password)
        issued, _ = await self._open_session(user, user_agent=user_agent)
        return issued

    async def refresh(self, token: str, *, user_agent: str = "") -> IssuedSession:
        """Обменять refresh-токен на новую пару, погасив предъявленный.

        Ротация обязательна: токен, которым уже обменялись, больше не работает, а
        предъявление такого токена повторно считается кражей и гасит всю цепочку.
        """
        session = await self._sessions.get_by_digest(self._refresh_tokens.digest(token))
        now = self._clock.now()
        if session is None:
            raise InvalidTokenError("Refresh-токен неизвестен.")
        if session.is_revoked:
            await self._sessions.revoke_family(session.family_id, revoked_at=now)
            raise RefreshTokenReuseError()
        if session.is_expired(now):
            await self._sessions.revoke(session.identifier, revoked_at=now)
            raise TokenExpiredError("Сессия истекла, войдите заново.")

        user = await self._users.get_by_id(session.user_id)
        if user is None or not user.is_active:
            await self._sessions.revoke_all_for_user(session.user_id, revoked_at=now)
            raise AccountInactiveError()

        issued, successor = await self._open_session(
            user,
            family_id=session.family_id,
            user_agent=user_agent or session.user_agent,
        )
        await self._sessions.revoke(
            session.identifier,
            revoked_at=now,
            replaced_by=successor.identifier,
        )
        return issued

    async def resolve_access_token(self, token: str) -> User:
        """Установить учётную запись по токену доступа.

        Роль берётся из записи, а не из токена: пониженный в правах администратор
        теряет их сразу, а не когда истечёт выданный ранее токен.
        """
        claims = self._access_tokens.read(token)
        user = await self._users.get_by_id(claims.subject)
        if user is None:
            raise InvalidTokenError("Учётной записи из токена больше нет.")
        if not user.is_active:
            raise AccountInactiveError()
        return user

    async def log_out(self, token: str | None) -> None:
        """Погасить сессию, которой соответствует предъявленный refresh-токен.

        Выход идемпотентен и молчит о том, был ли токен вообще: клиенту нечего
        из этого извлекать, а постороннему — тем более.
        """
        if not token:
            return
        session = await self._sessions.get_by_digest(self._refresh_tokens.digest(token))
        if session is None or session.is_revoked:
            return
        await self._sessions.revoke(session.identifier, revoked_at=self._clock.now())

    async def log_out_everywhere(self, user_id: str) -> int:
        """Погасить все сессии учётной записи и сообщить, сколько их было."""
        return await self._sessions.revoke_all_for_user(user_id, revoked_at=self._clock.now())

    async def _open_session(
        self,
        user: User,
        *,
        family_id: str | None = None,
        user_agent: str = "",
    ) -> tuple[IssuedSession, RefreshSession]:
        """Выдать пару токенов и запомнить породившую её сессию."""
        now = self._clock.now()
        minted = self._refresh_tokens.mint()
        session = RefreshSession(
            user_id=user.identifier,
            token_digest=minted.digest,
            family_id=family_id or uuid.uuid4().hex,
            issued_at=now,
            expires_at=now + timedelta(seconds=self._refresh_ttl_seconds),
            user_agent=user_agent,
        )
        await self._sessions.create(session)

        access_token = self._access_tokens.issue(user)
        issued = IssuedSession(
            user=user,
            access_token=access_token,
            refresh_token=minted.value,
            refresh_expires_at=session.expires_at,
            csrf_token=secrets.token_urlsafe(CSRF_TOKEN_BYTES),
            expires_in=self._access_ttl_seconds,
            refresh_expires_in=self._refresh_ttl_seconds,
        )
        return issued, session

    async def _upgrade_hash_if_stale(self, user: User, password: str) -> None:
        """Пересчитать хеш пароля, если параметры алгоритма с тех пор изменились."""
        if not self._hasher.needs_rehash(user.password_hash):
            return
        user.password_hash = self._hasher.hash(password)
        await self._users.update(user)


def _display_name_from_email(email: str) -> str:
    """Собрать подпись терминала из адреса почты."""
    return email.split("@", 1)[0][:64] or "выживший"
