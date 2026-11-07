"""Провайдеры внедрения зависимостей для слоя API.

Здесь строится каждый объект, нужный роутерам, и внедряется через ``Depends``:
клиент хранилища и репозитории живут столько же, сколько процесс, а RAG-сервис
получает те абстракции, которые объявляет.

Собирающие функции отделены от провайдеров ``Depends`` намеренно: офлайн-стенд
оценки строит те же зависимости без графа FastAPI, и общий код гарантирует, что
он оценивает боевой конвейер, а не его копию.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from typing import Annotated, Final

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer

from app.config import Settings, get_settings
from app.cookies import AuthCookies
from app.domain import PasswordPolicy, User
from app.exceptions import AIServiceError, AuthorizationError, InvalidTokenError
from app.interfaces import (
    AccessTokenIssuerProtocol,
    AIClientProtocol,
    CharacterRepositoryProtocol,
    ClockProtocol,
    LoreChunkRepositoryProtocol,
    PasswordHasherProtocol,
    QueryExpanderProtocol,
    RefreshSessionRepositoryProtocol,
    RefreshTokenFactoryProtocol,
    TracerProtocol,
    UserRepositoryProtocol,
)
from app.services import (
    AccessPolicy,
    ArchivePromptBuilder,
    Argon2PasswordHasher,
    AuthService,
    HideQueryExpander,
    JwtAccessTokenIssuer,
    NoQueryExpander,
    RAGService,
    RetrievalPolicy,
    SecureRefreshTokenFactory,
    SystemClock,
    TierPolicy,
)
from app.vector import (
    Bm25Encoder,
    QdrantCharacterRepository,
    QdrantLoreChunkRepository,
    QdrantRefreshSessionRepository,
    QdrantUserRepository,
    SparseEncoderProtocol,
    VectorStore,
)


async def get_vector_store(request: Request) -> VectorStore:
    """Вернуть хранилище, созданное жизненным циклом приложения."""
    store: VectorStore | None = getattr(request.app.state, "vector_store", None)
    if store is None:
        raise RuntimeError("Vector store is not initialised; the application lifespan did not run")
    return store


@lru_cache(maxsize=1)
def _cached_sparse_encoder(model_name: str, language: str) -> Bm25Encoder:
    """Собрать энкодер BM25 ровно один раз на процесс."""
    return Bm25Encoder(model_name, language=language)


def build_sparse_encoder(settings: Settings) -> SparseEncoderProtocol:
    """Вернуть энкодер BM25, общий для всего процесса.

    Кэш ключуется строками, а не объектом настроек, потому что ``Settings``
    неизменяем, но не хешируем: адресом ключа служит то, что действительно
    отличает один энкодер от другого, — модель и её язык.
    """
    return _cached_sparse_encoder(
        settings.sparse_embedding_model,
        settings.sparse_embedding_language,
    )


def build_character_repository(store: VectorStore) -> QdrantCharacterRepository:
    """Собрать репозиторий образов выживших поверх хранилища."""
    return QdrantCharacterRepository(store.client, store.character_collection)


def build_lore_chunk_repository(
    store: VectorStore,
    sparse_encoder: SparseEncoderProtocol,
    settings: Settings,
) -> QdrantLoreChunkRepository:
    """Собрать репозиторий лора вместе с настройками его гибридного поиска."""
    return QdrantLoreChunkRepository(
        store.client,
        store.lore_collection,
        sparse_encoder,
        hybrid_enabled=settings.rag_hybrid_enabled,
        prefetch_multiplier=settings.rag_prefetch_multiplier,
    )


def build_retrieval_policy(settings: Settings) -> RetrievalPolicy:
    """Собрать бюджет и пороги уровней доказательств из конфигурации."""
    return RetrievalPolicy(
        official=TierPolicy(
            limit=settings.rag_official_top_k,
            min_similarity=settings.rag_min_similarity,
        ),
        graffiti=TierPolicy(
            limit=settings.rag_graffiti_top_k,
            min_similarity=settings.rag_min_similarity,
        ),
        voice=TierPolicy(
            limit=settings.rag_voice_top_k,
            min_similarity=settings.rag_voice_min_similarity,
        ),
    )


def build_rag_service(
    character_repository: CharacterRepositoryProtocol,
    lore_chunk_repository: LoreChunkRepositoryProtocol,
    prompt_builder: ArchivePromptBuilder,
    ai_client: AIClientProtocol,
    tracer: TracerProtocol,
    settings: Settings,
    query_expander: QueryExpanderProtocol,
) -> RAGService:
    """Собрать RAG-конвейер из его зависимостей и конфигурации."""
    return RAGService(
        character_repository,
        lore_chunk_repository,
        prompt_builder,
        ai_client,
        policy=build_retrieval_policy(settings),
        query_expander=query_expander,
        tracer=tracer,
    )


async def get_ai_client(request: Request) -> AIClientProtocol:
    """Вернуть общий AI-клиент, созданный жизненным циклом приложения."""
    client: AIClientProtocol | None = getattr(request.app.state, "ai_client", None)
    if client is None:
        raise AIServiceError("AI client is not initialised; the application lifespan did not run")
    return client


async def get_tracer(request: Request) -> TracerProtocol:
    """Вернуть трейсер, созданный жизненным циклом приложения."""
    tracer: TracerProtocol | None = getattr(request.app.state, "tracer", None)
    if tracer is None:
        raise RuntimeError("Tracer is not initialised; the application lifespan did not run")
    return tracer


def get_character_repository(
    store: Annotated[VectorStore, Depends(get_vector_store)],
) -> QdrantCharacterRepository:
    """Привязать репозиторий образов к коллекции хранилища."""
    return build_character_repository(store)


def get_lore_chunk_repository(
    store: Annotated[VectorStore, Depends(get_vector_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> QdrantLoreChunkRepository:
    """Привязать репозиторий лора к коллекции хранилища и к его гибридным настройкам."""
    return build_lore_chunk_repository(store, build_sparse_encoder(settings), settings)


def get_prompt_builder() -> ArchivePromptBuilder:
    """Вернуть сборщик многоуровневого брифинга по архиву.

    Он не хранит состояния, так что одного экземпляра хватило бы на весь процесс;
    провайдер сохраняет единообразие графа зависимостей и позволяет тривиально
    подменить политику промптов из теста.
    """
    return ArchivePromptBuilder()


def build_query_expander(
    ai_client: AIClientProtocol,
    settings: Settings,
    tracer: TracerProtocol,
) -> QueryExpanderProtocol:
    """Собрать расширитель запроса по настройкам: HIDE или его выключенную заглушку.

    Отдельная функция нужна офлайн-стенду оценки: он собирает те же зависимости, что
    и HTTP-слой, но без графа ``Depends``.
    """
    if not settings.hide_enabled:
        return NoQueryExpander()
    return HideQueryExpander(
        ai_client,
        max_questions=settings.hide_max_questions,
        tracer=tracer,
    )


def get_query_expander(
    ai_client: Annotated[AIClientProtocol, Depends(get_ai_client)],
    settings: Annotated[Settings, Depends(get_settings)],
    tracer: Annotated[TracerProtocol, Depends(get_tracer)],
) -> QueryExpanderProtocol:
    """Вернуть расширитель запроса, общий для всех ходов разговора."""
    return build_query_expander(ai_client, settings, tracer)


def get_rag_service(
    character_repository: Annotated[QdrantCharacterRepository, Depends(get_character_repository)],
    lore_chunk_repository: Annotated[
        QdrantLoreChunkRepository, Depends(get_lore_chunk_repository)
    ],
    prompt_builder: Annotated[ArchivePromptBuilder, Depends(get_prompt_builder)],
    ai_client: Annotated[AIClientProtocol, Depends(get_ai_client)],
    tracer: Annotated[TracerProtocol, Depends(get_tracer)],
    settings: Annotated[Settings, Depends(get_settings)],
    query_expander: Annotated[QueryExpanderProtocol, Depends(get_query_expander)],
) -> RAGService:
    """Собрать RAG-конвейер из его зависимостей и конфигурации."""
    return build_rag_service(
        character_repository,
        lore_chunk_repository,
        prompt_builder,
        ai_client,
        tracer,
        settings,
        query_expander,
    )


VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
CharacterRepositoryDep = Annotated[QdrantCharacterRepository, Depends(get_character_repository)]
LoreChunkRepositoryDep = Annotated[
    QdrantLoreChunkRepository, Depends(get_lore_chunk_repository)
]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


# --- Authentication and authorization -------------------------------------


def get_clock() -> ClockProtocol:
    """Вернуть часы процесса: по ним считаются сроки жизни токенов и сессий."""
    return SystemClock()


def get_auth_cookies(settings: SettingsDep) -> AuthCookies:
    """Вернуть раскладку cookie, собранную из настроек безопасности."""
    return AuthCookies(settings)


@lru_cache(maxsize=1)
def build_password_hasher() -> Argon2PasswordHasher:
    """Вернуть хешер паролей, общий для всего процесса.

    Кэш здесь не оптимизация, а необходимость: сборка хешера считает одну
    строку-заглушку настоящим Argon2, то есть десятки миллисекунд, и платить их
    на каждом запросе незачем. Экземпляр не хранит состояния, поэтому делить его
    между запросами безопасно.
    """
    return Argon2PasswordHasher()


def build_access_token_issuer(settings: Settings, clock: ClockProtocol) -> JwtAccessTokenIssuer:
    """Собрать выдачу токенов доступа из настроек подписи."""
    return JwtAccessTokenIssuer(
        settings.auth_jwt_secret,
        algorithm=settings.auth_jwt_algorithm,
        issuer=settings.auth_issuer,
        audience=settings.auth_audience,
        ttl_seconds=settings.auth_access_token_ttl_seconds,
        clock=clock,
    )


def build_auth_service(
    users: UserRepositoryProtocol,
    sessions: RefreshSessionRepositoryProtocol,
    hasher: PasswordHasherProtocol,
    access_tokens: AccessTokenIssuerProtocol,
    refresh_tokens: RefreshTokenFactoryProtocol,
    clock: ClockProtocol,
    settings: Settings,
) -> AuthService:
    """Собрать сервис учётных записей из его зависимостей и конфигурации.

    Отдельная функция нужна инструментам вне графа ``Depends``: скрипту первого
    администратора, который заводит учётную запись теми же правилами, что и API.
    """
    return AuthService(
        users,
        sessions,
        hasher,
        access_tokens,
        refresh_tokens,
        clock,
        policy=PasswordPolicy(),
        access_ttl_seconds=settings.auth_access_token_ttl_seconds,
        refresh_ttl_seconds=settings.auth_refresh_token_ttl_seconds,
        registration_enabled=settings.auth_registration_enabled,
    )


def build_user_repository(store: VectorStore) -> QdrantUserRepository:
    """Собрать репозиторий учётных записей поверх хранилища."""
    return QdrantUserRepository(store.client, store.user_collection)


def build_refresh_session_repository(
    store: VectorStore,
) -> QdrantRefreshSessionRepository:
    """Собрать репозиторий refresh-сессий поверх хранилища."""
    return QdrantRefreshSessionRepository(store.client, store.refresh_session_collection)


def get_user_repository(store: VectorStoreDep) -> QdrantUserRepository:
    """Привязать репозиторий учётных записей к коллекции хранилища."""
    return build_user_repository(store)


def get_refresh_session_repository(
    store: VectorStoreDep,
) -> QdrantRefreshSessionRepository:
    """Привязать репозиторий сессий к коллекции хранилища."""
    return build_refresh_session_repository(store)


def get_access_token_issuer(
    settings: SettingsDep,
    clock: Annotated[ClockProtocol, Depends(get_clock)],
) -> JwtAccessTokenIssuer:
    """Вернуть выдачу токенов доступа, общую для всех запросов."""
    return build_access_token_issuer(settings, clock)


def get_password_hasher() -> PasswordHasherProtocol:
    """Вернуть хешер паролей, общий для всего процесса."""
    return build_password_hasher()


def build_refresh_token_factory() -> SecureRefreshTokenFactory:
    """Вернуть фабрику refresh-токенов: она не хранит состояния и детерминирована."""
    return SecureRefreshTokenFactory()


def get_access_policy() -> AccessPolicy:
    """Вернуть правила авторизации."""
    return AccessPolicy()


def get_auth_service(
    users: Annotated[UserRepositoryProtocol, Depends(get_user_repository)],
    sessions: Annotated[
        RefreshSessionRepositoryProtocol, Depends(get_refresh_session_repository)
    ],
    settings: SettingsDep,
    clock: Annotated[ClockProtocol, Depends(get_clock)],
    hasher: Annotated[PasswordHasherProtocol, Depends(get_password_hasher)],
    access_tokens: Annotated[AccessTokenIssuerProtocol, Depends(get_access_token_issuer)],
    refresh_tokens: Annotated[RefreshTokenFactoryProtocol, Depends(build_refresh_token_factory)],
) -> AuthService:
    """Собрать сервис учётных записей из его зависимостей и конфигурации.

    Каждый участник приходит сюда через ``Depends``, а не создаётся на месте:
    иначе тест не смог бы подменить часы или хешер, не трогая остальной граф.
    """
    return build_auth_service(
        users,
        sessions,
        hasher,
        access_tokens,
        refresh_tokens,
        clock,
        settings,
    )


OAUTH2_SCHEME: Final[OAuth2PasswordBearer] = OAuth2PasswordBearer(
    tokenUrl=f"{get_settings().api_prefix}/auth/token",
    auto_error=False,
)
"""Схема OAuth 2 (password grant) для документации и заголовка ``Authorization``.

Объект создаётся на импорте, потому что FastAPI описывает схему безопасности по
самому объекту в графе зависимостей, а не по вызову внутри провайдера: иначе
кнопка «Authorize» в документации осталась бы без объявленного потока.
``auto_error=False`` оставляет отказ этому приложению: истёкший токен и
отсутствующий токен — разные ответы, и подменять их стандартным «not authenticated»
нельзя.
"""


async def get_current_user(
    request: Request,
    header_token: Annotated[str | None, Depends(OAUTH2_SCHEME)],
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
    settings: SettingsDep,
) -> User:
    """Установить учётную запись, от чьего имени пришёл запрос.

    Cookie проверяется первой: браузер прикладывает её сам, и страница не может
    ни прочитать токен, ни подменить его. Заголовок нужен тем, кто не браузер.
    """
    token = request.cookies.get(settings.auth_access_cookie_name) or header_token
    if not token:
        raise InvalidTokenError("Запрос не предъявил токен доступа.")
    return await auth_service.resolve_access_token(token)


async def get_current_admin(
    user: Annotated[User, Depends(get_current_user)],
    policy: Annotated[AccessPolicy, Depends(get_access_policy)],
) -> User:
    """Потребовать административных прав у установленной учётной записи."""
    policy.ensure_admin(user)
    return user


async def require_csrf_token(request: Request, settings: SettingsDep) -> None:
    """Потребовать CSRF-метку от изменяющего запроса, аутентифицированного cookie.

    Браузер прикладывает cookie к запросу с чужого сайта сам, поэтому одной
    cookie для изменяющих запросов мало: клиент обязан повторить значение из
    читаемой cookie заголовком, а посторонний сайт подставить заголовок не может.

    Проверка не касается двух случаев. Запрос с токеном в заголовке
    ``Authorization``: его чужой сайт не отправит вовсе, подставлять нечего. И
    запрос без cookie сессии: подменять в нём тем более нечего, а ответить на него
    должна аутентификация — отказом «нет доступа», а не «нет метки».
    """
    if request.headers.get("Authorization", "").lower().startswith("bearer "):
        return
    if not any(
        request.cookies.get(name)
        for name in (settings.auth_access_cookie_name, settings.auth_refresh_cookie_name)
    ):
        return

    from_cookie = request.cookies.get(settings.auth_csrf_cookie_name, "")
    from_header = request.headers.get(settings.auth_csrf_header_name, "")
    if not from_cookie or not from_header or not secrets.compare_digest(from_cookie, from_header):
        raise AuthorizationError("CSRF-метка отсутствует или не совпадает.")


CurrentUserDep = Annotated[User, Depends(get_current_user)]
AdminUserDep = Annotated[User, Depends(get_current_admin)]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
AuthCookiesDep = Annotated[AuthCookies, Depends(get_auth_cookies)]
UserRepositoryDep = Annotated[QdrantUserRepository, Depends(get_user_repository)]
RefreshSessionRepositoryDep = Annotated[
    QdrantRefreshSessionRepository, Depends(get_refresh_session_repository)
]
AccessPolicyDep = Annotated[AccessPolicy, Depends(get_access_policy)]
ClockDep = Annotated[ClockProtocol, Depends(get_clock)]
