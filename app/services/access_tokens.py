from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from app.domain import User
from app.exceptions import InvalidTokenError, TokenExpiredError
from app.interfaces import AccessTokenClaims, ClockProtocol, IssuedAccessToken


class JwtAccessTokenIssuer:
    """Токены доступа в виде подписанных JWT.

    Токен самодостаточен: издателя, получателя и срок жизни он несёт в себе, а
    подпись отвечает за то, что его не подделали. Отозвать выданный токен нельзя —
    поэтому срок жизни короткий, а всё, что можно отозвать, живёт в refresh-сессии.
    """

    def __init__(
        self,
        secret: str,
        *,
        algorithm: str,
        issuer: str,
        audience: str,
        ttl_seconds: int,
        clock: ClockProtocol,
    ) -> None:
        self._secret = secret
        self._algorithm = algorithm
        self._issuer = issuer
        self._audience = audience
        self._ttl_seconds = ttl_seconds
        self._clock = clock

    def issue(self, user: User) -> IssuedAccessToken:
        """Подписать токен доступа, удостоверяющий эту учётную запись."""
        now = self._clock.now()
        expires_at = now + timedelta(seconds=self._ttl_seconds)
        payload: dict[str, Any] = {
            "sub": user.identifier,
            "email": user.email,
            "role": str(user.role),
            # Утверждения нужны клиенту и трейсингу, но не проверке прав: права
            # читаются из учётной записи, поэтому смена роли действует сразу,
            # а не после того, как истечёт выданный ранее токен.
            "iat": now,
            "exp": expires_at,
            "iss": self._issuer,
            "aud": self._audience,
            "jti": uuid.uuid4().hex,
        }
        return IssuedAccessToken(
            value=jwt.encode(payload, self._secret, algorithm=self._algorithm),
            expires_at=expires_at,
        )

    def read(self, token: str) -> AccessTokenClaims:
        """Разобрать токен и вернуть его содержимое.

        Срок жизни сверяется часами сервиса, а не часами библиотеки: время —
        зависимость этого приложения, и оно обязано быть одним и тем же при
        выдаче и при проверке. Иначе токен, выданный по часам приложения,
        отвергался бы по часам процесса, и разницу нельзя было бы ни объяснить,
        ни проверить тестом.
        """
        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=[self._algorithm],
                audience=self._audience,
                issuer=self._issuer,
                options={
                    "require": ["exp", "iat", "sub"],
                    "verify_exp": False,
                    "verify_iat": False,
                },
            )
            expires_at = _to_datetime(payload["exp"])
            issued_at = _to_datetime(payload["iat"])
        except jwt.InvalidTokenError as exc:
            raise InvalidTokenError("Токен доступа недействителен.") from exc
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise InvalidTokenError("Токен доступа недействителен.") from exc

        if self._clock.now() >= expires_at:
            raise TokenExpiredError("Срок действия токена доступа истёк.")

        return AccessTokenClaims(
            subject=str(payload["sub"]),
            email=str(payload.get("email", "")),
            role=str(payload.get("role", "")),
            issued_at=issued_at,
            expires_at=expires_at,
        )


def _to_datetime(claim: Any) -> datetime:
    """Преобразовать числовое утверждение времени в момент UTC."""
    return datetime.fromtimestamp(float(claim), tz=UTC)
