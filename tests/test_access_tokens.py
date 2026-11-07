"""Проверки токенов: подпись, срок жизни, издатель, получатель и отпечатки."""

from __future__ import annotations

import jwt
import pytest

from app.exceptions import InvalidTokenError, TokenExpiredError
from app.services import JwtAccessTokenIssuer, SecureRefreshTokenFactory
from tests.fakes import FixedClock, build_user

SECRET = "s" * 48
USER = build_user(user_id="user-7", email="ellis@l4d.dev")


def build_issuer(
    clock: FixedClock,
    *,
    secret: str = SECRET,
    issuer: str = "l4d",
    audience: str = "terminal",
    ttl_seconds: int = 900,
) -> JwtAccessTokenIssuer:
    return JwtAccessTokenIssuer(
        secret,
        algorithm="HS256",
        issuer=issuer,
        audience=audience,
        ttl_seconds=ttl_seconds,
        clock=clock,
    )


def test_a_fresh_token_carries_the_identity_and_the_expiry() -> None:
    clock = FixedClock()
    issued = build_issuer(clock, ttl_seconds=900).issue(USER)

    claims = build_issuer(clock).read(issued.value)

    assert claims.subject == "user-7"
    assert claims.email == "ellis@l4d.dev"
    assert claims.role == "member"
    assert claims.issued_at == clock.now()
    assert claims.expires_at == issued.expires_at
    assert (issued.expires_at - clock.now()).total_seconds() == 900


def test_an_expired_token_is_refused_for_being_expired() -> None:
    """Истечение — отдельный ответ: клиенту сказано обновить сессию, а не входить заново."""
    clock = FixedClock()
    issuer = build_issuer(clock, ttl_seconds=60)
    token = issuer.issue(USER).value

    clock.advance(61)

    with pytest.raises(TokenExpiredError):
        issuer.read(token)


def test_a_token_signed_by_someone_else_is_refused() -> None:
    foreign = build_issuer(FixedClock(), secret="x" * 48).issue(USER)

    with pytest.raises(InvalidTokenError):
        build_issuer(FixedClock()).read(foreign.value)


def test_a_token_meant_for_another_audience_is_refused() -> None:
    """Получатель проверяется: чужой сервис не должен пускать со своим токеном."""
    foreign = build_issuer(FixedClock(), audience="another-service").issue(USER)

    with pytest.raises(InvalidTokenError):
        build_issuer(FixedClock()).read(foreign.value)


def test_a_token_without_a_subject_is_refused() -> None:
    """Токен без субъекта не называет учётную запись, поэтому не годится вовсе."""
    clock = FixedClock()
    payload = {"iat": clock.now(), "exp": clock.now(), "iss": "l4d", "aud": "terminal"}
    token = jwt.encode(payload, SECRET, algorithm="HS256")

    with pytest.raises(InvalidTokenError):
        build_issuer(clock).read(token)


def test_a_tampered_token_is_refused() -> None:
    issuer = build_issuer(FixedClock())
    token = issuer.issue(USER).value

    with pytest.raises(InvalidTokenError):
        issuer.read(f"{token[:-4]}zzzz")


def test_refresh_tokens_are_random_and_hashable() -> None:
    """Отпечаток обязан быть устойчивым: по нему сессию и находят, и сверяют."""
    factory = SecureRefreshTokenFactory()

    first = factory.mint()
    second = factory.mint()

    assert first.value != second.value
    assert first.digest == factory.digest(first.value)
    assert first.digest != second.digest
    assert len(first.digest) == 64


def test_a_hundred_refresh_tokens_do_not_repeat() -> None:
    factory = SecureRefreshTokenFactory()

    values = {factory.mint().value for _ in range(100)}

    assert len(values) == 100
