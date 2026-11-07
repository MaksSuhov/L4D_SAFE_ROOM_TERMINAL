"""Интеграционные тесты аутентификации и авторизации через HTTP.

Подменяются только хранилище учётных записей, хешер и часы; роутеры, граф
зависимостей, cookie-слой и сервис учётных записей работают по-настоящему.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.dependencies import (
    get_ai_client,
    get_character_repository,
    get_lore_chunk_repository,
    get_query_expander,
    get_vector_store,
)
from app.domain import User
from app.exceptions import (
    AccountInactiveError,
    AuthorizationError,
    InvalidCredentialsError,
    InvalidTokenError,
    RefreshTokenReuseError,
    TokenExpiredError,
)
from app.main import create_app
from app.taxonomy import UserRole
from tests.auth_support import (
    ACCESS_COOKIE,
    CSRF_COOKIE,
    CSRF_HEADER,
    REFRESH_COOKIE,
    REFRESH_COOKIE_PATH,
    log_in,
    override_auth_dependencies,
    register,
)
from tests.fakes import (
    FakeAIClient,
    FakePasswordHasher,
    FakeQueryExpander,
    FakeVectorStore,
    FixedClock,
    InMemoryCharacterRepository,
    InMemoryLoreChunkRepository,
    InMemoryRefreshSessionRepository,
    InMemoryUserRepository,
    build_character,
    build_user,
)

API_PREFIX = get_settings().api_prefix
ACCESS_TTL = get_settings().auth_access_token_ttl_seconds

PASSWORD = "safe-room-passphrase"
ADMIN_EMAIL = "operator@l4d.dev"
MEMBER_EMAIL = "ellis@l4d.dev"

ADMIN_ID = "user-operator"
MEMBER_ID = "user-ellis"


def build_admin() -> User:
    """Завести оператора заново: учётные записи в тестах меняются, делиться ими нельзя."""
    return build_user(
        user_id=ADMIN_ID,
        email=ADMIN_EMAIL,
        password_hash=f"fake${PASSWORD}",
        display_name="Оператор",
        role=UserRole.ADMIN,
    )


def build_member() -> User:
    """Завести выжившего заново по той же причине."""
    return build_user(
        user_id=MEMBER_ID,
        email=MEMBER_EMAIL,
        password_hash=f"fake${PASSWORD}",
        display_name="Эллис",
    )


@pytest.fixture
def users() -> InMemoryUserRepository:
    """Хранилище учётных записей с одним оператором и одним выжившим."""
    return InMemoryUserRepository([build_admin(), build_member()])


@pytest.fixture
def application(
    users: InMemoryUserRepository,
    sessions: InMemoryRefreshSessionRepository,
    hasher: FakePasswordHasher,
    clock: FixedClock,
) -> Iterator[FastAPI]:
    """Приложение с двойниками вместо хранилища, часов и хешера паролей."""
    application = create_app()
    override_auth_dependencies(
        application,
        users=users,
        sessions=sessions,
        hasher=hasher,
        clock=clock,
    )
    application.dependency_overrides[get_character_repository] = lambda: (
        InMemoryCharacterRepository([build_character(character_id=1, name="Эллис")])
    )
    application.dependency_overrides[get_lore_chunk_repository] = lambda: (
        InMemoryLoreChunkRepository()
    )
    application.dependency_overrides[get_ai_client] = lambda: FakeAIClient()
    application.dependency_overrides[get_query_expander] = lambda: FakeQueryExpander()
    application.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()

    yield application
    application.dependency_overrides.clear()


@pytest.fixture
def client(application: FastAPI) -> Iterator[TestClient]:
    """Клиент без сессии: cookie он получает только тем запросом, который сам войдёт."""
    with TestClient(application) as test_client:
        yield test_client


def raw_client(application: FastAPI) -> TestClient:
    """Клиент, который не хранит cookie: так проверяются токены из тела запроса."""
    return TestClient(application)


def test_registration_opens_a_session_in_cookies(client: TestClient) -> None:
    response = client.post(
        f"{API_PREFIX}/auth/register",
        json={"email": "rookie@l4d.dev", "password": PASSWORD},
    )

    assert response.status_code == 201
    cookies = response.headers.get_list("set-cookie")
    access_cookie = next(c for c in cookies if c.startswith(f"{ACCESS_COOKIE}="))
    assert "HttpOnly" in access_cookie
    assert f"Path={REFRESH_COOKIE_PATH}" not in access_cookie
    refresh_cookie = next(c for c in cookies if c.startswith(f"{REFRESH_COOKIE}="))
    assert "HttpOnly" in refresh_cookie
    assert f"Path={REFRESH_COOKIE_PATH}" in refresh_cookie
    csrf_cookie = next(c for c in cookies if c.startswith(f"{CSRF_COOKIE}="))
    assert "HttpOnly" not in csrf_cookie


def test_registration_returns_the_member_and_the_csrf_token(client: TestClient) -> None:
    payload = register(
        client,
        prefix=API_PREFIX,
        email="rookie@l4d.dev",
        password=PASSWORD,
        display_name="Новичок",
    )

    assert payload["user"]["email"] == "rookie@l4d.dev"
    assert payload["user"]["display_name"] == "Новичок"
    assert payload["user"]["role"] == "member"
    assert payload["user"]["is_active"] is True
    assert payload["expires_in"] == ACCESS_TTL
    assert payload["csrf_token"]


def test_registration_does_not_leak_the_password_hash(client: TestClient) -> None:
    response = client.post(
        f"{API_PREFIX}/auth/register",
        json={"email": "rookie@l4d.dev", "password": PASSWORD},
    )

    assert "password" not in response.text


def test_a_taken_email_is_refused_with_a_conflict(client: TestClient) -> None:
    response = client.post(
        f"{API_PREFIX}/auth/register",
        json={"email": "ELLIS@l4d.dev", "password": PASSWORD},
    )

    assert response.status_code == 409
    assert response.json()["code"] == "email_taken"


def test_a_weak_password_is_refused_as_a_validation_error(client: TestClient) -> None:
    response = client.post(
        f"{API_PREFIX}/auth/register",
        json={"email": "rookie@l4d.dev", "password": "short"},
    )

    assert response.status_code == 422


def test_a_password_from_the_blocklist_is_refused_by_the_policy(client: TestClient) -> None:
    response = client.post(
        f"{API_PREFIX}/auth/register",
        json={"email": "rookie@l4d.dev", "password": "password123"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "weak_password"


def test_an_invalid_email_is_refused(client: TestClient) -> None:
    response = client.post(
        f"{API_PREFIX}/auth/register",
        json={"email": "not-an-email", "password": PASSWORD},
    )

    assert response.status_code == 422


def test_registration_can_be_closed_by_configuration(
    users: InMemoryUserRepository,
    sessions: InMemoryRefreshSessionRepository,
    hasher: FakePasswordHasher,
    clock: FixedClock,
) -> None:
    settings = Settings(auth_registration_enabled=False)
    application = create_app(settings)
    override_auth_dependencies(
        application,
        users=users,
        sessions=sessions,
        hasher=hasher,
        clock=clock,
    )
    application.dependency_overrides[get_settings] = lambda: settings

    with TestClient(application) as closed:
        response = closed.post(
            f"{API_PREFIX}/auth/register",
            json={"email": "rookie@l4d.dev", "password": PASSWORD},
        )

    application.dependency_overrides.clear()
    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"


def test_the_token_endpoint_accepts_the_oauth2_form(client: TestClient) -> None:
    response = client.post(
        f"{API_PREFIX}/auth/token",
        data={"username": MEMBER_EMAIL, "password": PASSWORD},
    )

    assert response.status_code == 200
    assert response.json()["user"]["email"] == MEMBER_EMAIL
    assert ACCESS_COOKIE in client.cookies
    assert CSRF_COOKIE in client.cookies


def test_the_token_endpoint_renames_a_bad_password(client: TestClient) -> None:
    response = client.post(
        f"{API_PREFIX}/auth/token",
        data={"username": MEMBER_EMAIL, "password": "not-the-passphrase"},
    )

    assert response.status_code == 401
    assert response.json()["code"] == "invalid_credentials"
    assert response.headers["www-authenticate"] == "Bearer"


def test_the_token_endpoint_does_not_reveal_whether_the_email_exists(
    client: TestClient,
) -> None:
    known = client.post(
        f"{API_PREFIX}/auth/token",
        data={"username": MEMBER_EMAIL, "password": "not-the-passphrase"},
    )
    unknown = client.post(
        f"{API_PREFIX}/auth/token",
        data={"username": "nobody@l4d.dev", "password": "not-the-passphrase"},
    )

    assert known.status_code == unknown.status_code == 401
    assert known.json() == unknown.json()


def test_the_openapi_document_declares_the_password_flow(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    flows = schema["components"]["securitySchemes"]["OAuth2PasswordBearer"]["flows"]
    assert flows["password"]["tokenUrl"] == f"{API_PREFIX}/auth/token"
    assert schema["paths"][f"{API_PREFIX}/chat"]["post"]["security"]


def test_the_current_user_is_returned_to_the_session_holder(client: TestClient) -> None:
    log_in(client, prefix=API_PREFIX, email=MEMBER_EMAIL, password=PASSWORD)

    response = client.get(f"{API_PREFIX}/auth/me")

    assert response.status_code == 200
    assert response.json()["email"] == MEMBER_EMAIL
    assert response.json()["role"] == "member"


def test_a_request_without_a_session_is_refused(client: TestClient) -> None:
    response = client.get(f"{API_PREFIX}/auth/me")

    assert response.status_code == 401
    assert response.json()["code"] == "invalid_token"
    assert response.headers["www-authenticate"] == "Bearer"


def test_a_bearer_token_works_as_well_as_a_cookie(application: FastAPI) -> None:
    """Программному клиенту cookie не нужны: он называет токен сам."""
    with raw_client(application) as anonymous:
        response = anonymous.post(
            f"{API_PREFIX}/auth/token",
            data={"username": MEMBER_EMAIL, "password": PASSWORD},
        )
        token = anonymous.cookies[ACCESS_COOKIE]

    with raw_client(application) as programmatic:
        response = programmatic.get(
            f"{API_PREFIX}/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    assert response.json()["email"] == MEMBER_EMAIL


def test_an_expired_access_token_says_so(client: TestClient, clock: FixedClock) -> None:
    """Разный ответ на «истёк» и «не тот» — это то, по чему клиент решает обновляться."""
    log_in(client, prefix=API_PREFIX, email=MEMBER_EMAIL, password=PASSWORD)

    clock.advance(ACCESS_TTL + 1)

    response = client.get(f"{API_PREFIX}/auth/me")

    assert response.status_code == 401
    assert response.json()["code"] == "expired_token"


def test_blocking_an_account_takes_effect_before_the_token_expires(
    client: TestClient,
    users: InMemoryUserRepository,
) -> None:
    log_in(client, prefix=API_PREFIX, email=MEMBER_EMAIL, password=PASSWORD)
    account = users.users[MEMBER_EMAIL]
    account.is_active = False

    response = client.get(f"{API_PREFIX}/auth/me")

    assert response.status_code == 401
    assert response.json()["code"] == "account_inactive"


def test_refresh_rotates_the_refresh_cookie(client: TestClient) -> None:
    log_in(client, prefix=API_PREFIX, email=MEMBER_EMAIL, password=PASSWORD)
    before = client.cookies[REFRESH_COOKIE]

    response = client.post(f"{API_PREFIX}/auth/refresh")

    assert response.status_code == 200
    assert client.cookies[REFRESH_COOKIE] != before


def test_refresh_accepts_the_token_in_the_body(application: FastAPI) -> None:
    with raw_client(application) as op:
        payload = log_in(op, prefix=API_PREFIX, email=MEMBER_EMAIL, password=PASSWORD)
        refresh_token = op.cookies[REFRESH_COOKIE]

    assert payload["csrf_token"]
    with raw_client(application) as programmatic:
        response = programmatic.post(
            f"{API_PREFIX}/auth/refresh",
            json={"refresh_token": refresh_token},
        )

    assert response.status_code == 200


def test_refresh_without_a_token_is_refused(client: TestClient) -> None:
    response = client.post(f"{API_PREFIX}/auth/refresh")

    assert response.status_code == 401
    assert response.json()["code"] == "invalid_token"


def test_renewing_an_unknown_token_is_refused(application: FastAPI) -> None:
    with raw_client(application) as anonymous:
        response = anonymous.post(
            f"{API_PREFIX}/auth/refresh",
            json={"refresh_token": "a-token-nobody-issued"},
        )

    assert response.status_code == 401
    assert response.json()["code"] == "invalid_token"


def test_a_replayed_refresh_token_retires_the_whole_chain(application: FastAPI) -> None:
    """Повторное предъявление — признак кражи, и цепочка ротаций гасится целиком."""
    with raw_client(application) as victim:
        log_in(victim, prefix=API_PREFIX, email=MEMBER_EMAIL, password=PASSWORD)
        first = victim.cookies[REFRESH_COOKIE]
        victim.post(f"{API_PREFIX}/auth/refresh")
        second = victim.cookies[REFRESH_COOKIE]

    with raw_client(application) as thief:
        replay = thief.post(f"{API_PREFIX}/auth/refresh", json={"refresh_token": first})
        after = thief.post(f"{API_PREFIX}/auth/refresh", json={"refresh_token": second})

    assert replay.status_code == 401
    assert replay.json()["code"] == "refresh_token_reused"
    assert after.status_code == 401
    assert after.json()["code"] == "refresh_token_reused"


def test_an_expired_session_cannot_be_renewed(client: TestClient, clock: FixedClock) -> None:
    log_in(client, prefix=API_PREFIX, email=MEMBER_EMAIL, password=PASSWORD)
    clock.advance(get_settings().auth_refresh_token_ttl_seconds + 1)

    response = client.post(f"{API_PREFIX}/auth/refresh")

    assert response.status_code == 401
    assert response.json()["code"] == "expired_token"


def test_log_out_clears_the_cookies_and_retires_the_session(client: TestClient) -> None:
    log_in(client, prefix=API_PREFIX, email=MEMBER_EMAIL, password=PASSWORD)

    response = client.post(f"{API_PREFIX}/auth/logout")

    assert response.status_code == 204
    assert ACCESS_COOKIE not in client.cookies
    assert REFRESH_COOKIE not in client.cookies
    assert CSRF_COOKIE not in client.cookies
    assert client.post(f"{API_PREFIX}/auth/refresh").status_code == 401


def test_log_out_requires_the_csrf_mark(client: TestClient) -> None:
    log_in(client, prefix=API_PREFIX, email=MEMBER_EMAIL, password=PASSWORD)
    del client.headers[CSRF_HEADER]

    response = client.post(f"{API_PREFIX}/auth/logout")

    assert response.status_code == 403


def test_log_out_everywhere_reports_the_sessions_it_retired(client: TestClient) -> None:
    log_in(client, prefix=API_PREFIX, email=MEMBER_EMAIL, password=PASSWORD)

    response = client.post(f"{API_PREFIX}/auth/logout-all")

    assert response.status_code == 200
    assert response.json()["revoked"] == 1


def test_a_cookie_session_needs_the_csrf_mark_for_a_changing_request(
    client: TestClient,
) -> None:
    """Чужой сайт приложит cookie сам, а заголовок подставить не сможет."""
    log_in(client, prefix=API_PREFIX, email=MEMBER_EMAIL, password=PASSWORD)
    del client.headers[CSRF_HEADER]

    response = client.post(f"{API_PREFIX}/chat", json={"character_id": 1, "message": "Привет"})

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"


def test_the_wrong_csrf_mark_is_refused(client: TestClient) -> None:
    log_in(client, prefix=API_PREFIX, email=MEMBER_EMAIL, password=PASSWORD)
    client.headers[CSRF_HEADER] = "a-mark-from-another-session"

    response = client.post(f"{API_PREFIX}/chat", json={"character_id": 1, "message": "Привет"})

    assert response.status_code == 403


def test_a_cookie_session_passes_with_the_csrf_mark(client: TestClient) -> None:
    log_in(client, prefix=API_PREFIX, email=MEMBER_EMAIL, password=PASSWORD)

    response = client.post(f"{API_PREFIX}/chat", json={"character_id": 1, "message": "Привет"})

    assert response.status_code == 200


def test_a_bearer_session_needs_no_csrf_mark(application: FastAPI) -> None:
    """Заголовок Authorization чужой сайт не отправит, поэтому подстановки не будет."""
    with raw_client(application) as programmatic:
        programmatic.post(
            f"{API_PREFIX}/auth/token",
            data={"username": MEMBER_EMAIL, "password": PASSWORD},
        )
        token = programmatic.cookies[ACCESS_COOKIE]
        response = programmatic.post(
            f"{API_PREFIX}/chat",
            json={"character_id": 1, "message": "Привет"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200


def test_the_archive_is_closed_to_anonymous_visitors(client: TestClient) -> None:
    assert client.get(f"{API_PREFIX}/characters").status_code == 401


def test_a_member_cannot_administer_accounts(client: TestClient) -> None:
    log_in(client, prefix=API_PREFIX, email=MEMBER_EMAIL, password=PASSWORD)

    response = client.get(f"{API_PREFIX}/admin/users")

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"


def test_an_admin_lists_the_accounts(client: TestClient) -> None:
    log_in(client, prefix=API_PREFIX, email=ADMIN_EMAIL, password=PASSWORD)

    response = client.get(f"{API_PREFIX}/admin/users")

    assert response.status_code == 200
    assert [user["email"] for user in response.json()] == [MEMBER_EMAIL, ADMIN_EMAIL]


def test_an_admin_promotes_a_member_and_it_works_at_once(client: TestClient) -> None:
    log_in(client, prefix=API_PREFIX, email=ADMIN_EMAIL, password=PASSWORD)

    response = client.patch(
        f"{API_PREFIX}/admin/users/{MEMBER_ID}",
        json={"role": "admin"},
    )

    assert response.status_code == 200
    assert response.json()["role"] == "admin"


def test_an_admin_blocks_an_account(client: TestClient) -> None:
    log_in(client, prefix=API_PREFIX, email=ADMIN_EMAIL, password=PASSWORD)

    response = client.patch(f"{API_PREFIX}/admin/users/{MEMBER_ID}", json={"is_active": False})

    assert response.status_code == 200
    assert response.json()["is_active"] is False


def test_an_empty_change_is_refused(client: TestClient) -> None:
    log_in(client, prefix=API_PREFIX, email=ADMIN_EMAIL, password=PASSWORD)

    response = client.patch(f"{API_PREFIX}/admin/users/{MEMBER_ID}", json={})

    assert response.status_code == 422


def test_an_admin_cannot_change_their_own_record(client: TestClient) -> None:
    """Иначе терминал может остаться без оператора, и войти будет некому."""
    log_in(client, prefix=API_PREFIX, email=ADMIN_EMAIL, password=PASSWORD)

    response = client.patch(f"{API_PREFIX}/admin/users/{ADMIN_ID}", json={"role": "member"})

    assert response.status_code == 403


def test_an_admin_cannot_delete_themselves(client: TestClient) -> None:
    log_in(client, prefix=API_PREFIX, email=ADMIN_EMAIL, password=PASSWORD)

    response = client.delete(f"{API_PREFIX}/admin/users/{ADMIN_ID}")

    assert response.status_code == 403


def test_an_admin_deletes_another_account(client: TestClient) -> None:
    log_in(client, prefix=API_PREFIX, email=ADMIN_EMAIL, password=PASSWORD)

    response = client.delete(f"{API_PREFIX}/admin/users/{MEMBER_ID}")
    listing = client.get(f"{API_PREFIX}/admin/users")

    assert response.status_code == 204
    assert [user["email"] for user in listing.json()] == [ADMIN_EMAIL]


def test_an_unknown_account_is_not_found(client: TestClient) -> None:
    log_in(client, prefix=API_PREFIX, email=ADMIN_EMAIL, password=PASSWORD)

    response = client.patch(f"{API_PREFIX}/admin/users/user-nobody", json={"role": "admin"})

    assert response.status_code == 404


def test_an_admin_retires_the_sessions_of_a_member(
    client: TestClient,
    application: FastAPI,
) -> None:
    """Оператор гасит сессии другого человека, и тот больше не обновляет токен."""
    log_in(client, prefix=API_PREFIX, email=ADMIN_EMAIL, password=PASSWORD)
    with raw_client(application) as member_session:
        log_in(member_session, prefix=API_PREFIX, email=MEMBER_EMAIL, password=PASSWORD)

        response = client.post(f"{API_PREFIX}/admin/users/{MEMBER_ID}/sessions/revoke")
        renewed = member_session.post(f"{API_PREFIX}/auth/refresh")

    assert response.status_code == 200
    assert response.json()["revoked"] == 1
    assert renewed.status_code == 401


def test_admin_mutations_need_the_csrf_mark(client: TestClient) -> None:
    log_in(client, prefix=API_PREFIX, email=ADMIN_EMAIL, password=PASSWORD)
    del client.headers[CSRF_HEADER]

    response = client.delete(f"{API_PREFIX}/admin/users/{MEMBER_ID}")

    assert response.status_code == 403


def test_an_invalid_token_is_reported_as_such(client: TestClient) -> None:
    """Порчу токена клиент видит тем же кодом, что и его отсутствие."""
    client.cookies.set(ACCESS_COOKIE, "not-a-token", domain="testserver", path="/")

    response = client.get(f"{API_PREFIX}/auth/me")

    assert response.status_code == 401
    assert response.json()["code"] == "invalid_token"


def test_the_service_refuses_to_sign_without_a_secret(
    application: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Без ключа подписи терминал не стартует — и это проверяется на старте."""
    empty = Settings(auth_jwt_secret="")
    assert empty.is_auth_configured is False

    with pytest.raises(RuntimeError), TestClient(create_app(empty)) as broken:
        broken.get("/")


def test_the_settings_ask_for_both_security_flags(
    application: FastAPI,
) -> None:
    """SameSite=None без Secure браузер отвергнет, поэтому такое сочетание запрещено."""
    with pytest.raises(ValueError):
        Settings(auth_cookie_samesite="none", auth_cookie_secure=False)


def test_the_application_refuses_a_short_signing_secret() -> None:
    with pytest.raises(ValueError):
        Settings(auth_jwt_secret="too-short")


def test_access_errors_carry_machine_readable_codes() -> None:
    """Код нужен клиенту: по нему он отличает обновление сессии от повторного входа."""
    assert InvalidTokenError().code == "invalid_token"
    assert TokenExpiredError().code == "expired_token"
    assert InvalidCredentialsError().code == "invalid_credentials"
    assert AccountInactiveError().code == "account_inactive"
    assert RefreshTokenReuseError().code == "refresh_token_reused"
    assert AuthorizationError().code == "forbidden"
