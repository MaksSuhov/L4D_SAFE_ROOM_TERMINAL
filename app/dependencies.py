"""Провайдеры внедрения зависимостей для слоя API.

Здесь строится каждый объект, нужный роутерам, и внедряется через ``Depends``:
клиент хранилища и репозитории живут столько же, сколько процесс, а RAG-сервис
получает те абстракции, которые объявляет.

Собирающие функции отделены от провайдеров ``Depends`` намеренно: офлайн-стенд
оценки строит те же зависимости без графа FastAPI, и общий код гарантирует, что
он оценивает боевой конвейер, а не его копию.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.exceptions import AIServiceError
from app.interfaces import (
    AIClientProtocol,
    CharacterRepositoryProtocol,
    LoreChunkRepositoryProtocol,
    QueryExpanderProtocol,
    TracerProtocol,
)
from app.services import (
    ArchivePromptBuilder,
    HideQueryExpander,
    NoQueryExpander,
    RAGService,
    RetrievalPolicy,
    TierPolicy,
)
from app.vector import (
    Bm25Encoder,
    QdrantCharacterRepository,
    QdrantLoreChunkRepository,
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
