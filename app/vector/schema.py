"""Схема коллекций Qdrant и её идемпотентное создание.

Оба плеча гибридного поиска живут в одной точке: косинусный эмбеддинг и
разреженный BM25 того же текста. Держать их врозь нельзя — тогда фильтр по
уровню доказательств пришлось бы повторять в двух запросах и склеивать выдачу на
клиенте, а у Qdrant для этого есть prefetch.

Три коллекции из четырёх обходятся без векторов вовсе: образы, учётные записи и
сессии — это строки, которые ищут по полю, а не по близости.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from qdrant_client import AsyncQdrantClient, models

DENSE_VECTOR_NAME: Final[str] = "dense"
"""Имя плотного вектора: косинусный эмбеддинг провайдера."""

SPARSE_VECTOR_NAME: Final[str] = "bm25"
"""Имя разреженного вектора: лексическое плечо гибридного поиска."""

LORE_PAYLOAD_INDEXES: Final[tuple[tuple[str, models.PayloadSchemaType], ...]] = (
    ("source_type", models.PayloadSchemaType.KEYWORD),
    ("topic", models.PayloadSchemaType.KEYWORD),
    ("character_name", models.PayloadSchemaType.KEYWORD),
    ("content_hash", models.PayloadSchemaType.KEYWORD),
)
"""Поля архива, по которым выборка фильтрует: уровень, тема, персонаж и хеш."""

CHARACTER_PAYLOAD_INDEXES: Final[tuple[tuple[str, models.PayloadSchemaType], ...]] = (
    ("name", models.PayloadSchemaType.KEYWORD),
    ("id", models.PayloadSchemaType.INTEGER),
)
"""Поля образа, по которым его ищут: уникальное имя и выданный идентификатор."""

USER_PAYLOAD_INDEXES: Final[tuple[tuple[str, models.PayloadSchemaType], ...]] = (
    ("email", models.PayloadSchemaType.KEYWORD),
    ("role", models.PayloadSchemaType.KEYWORD),
    ("is_active", models.PayloadSchemaType.BOOL),
    ("created_at", models.PayloadSchemaType.DATETIME),
)
"""Поля учётной записи: адрес почты как логин и признаки для отчётности."""

REFRESH_SESSION_PAYLOAD_INDEXES: Final[tuple[tuple[str, models.PayloadSchemaType], ...]] = (
    ("token_digest", models.PayloadSchemaType.KEYWORD),
    ("user_id", models.PayloadSchemaType.KEYWORD),
    ("family_id", models.PayloadSchemaType.KEYWORD),
    ("expires_at", models.PayloadSchemaType.DATETIME),
)
"""Поля сессии: по отпечатку её находят, по цепочке и владельцу — гасят пачкой."""


async def ensure_lore_collection(
    client: AsyncQdrantClient,
    collection: str,
    *,
    dimensions: int,
    recreate: bool = False,
) -> None:
    """Создать коллекцию архива вместе с payload-индексами.

    Повторный вызов безвреден: коллекция и индексы создаются только тогда, когда
    их ещё нет, поэтому наполнение можно запускать сколько угодно раз.
    """
    if recreate:
        await client.delete_collection(collection)

    if not await client.collection_exists(collection):
        await client.create_collection(
            collection_name=collection,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dimensions,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                # IDF считает сервер по статистике коллекции, поэтому энкодер
                # кладёт в вектор частоты терминов, а не их вес: запись, впервые
                # увиденная на запросе, всё равно получает корректный вклад.
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
        )

    for field_name, field_schema in LORE_PAYLOAD_INDEXES:
        await client.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=field_schema,
        )


async def ensure_character_collection(
    client: AsyncQdrantClient,
    collection: str,
    *,
    recreate: bool = False,
) -> None:
    """Создать коллекцию образов выживших.

    Векторов у неё нет вовсе: образ — это текст промпта и примеры реплик, их
    никто не ищет по близости. Коллекция нужна как хранилище, и она избавляет
    проект от реляционной базы ради считаных строк.
    """
    await _ensure_payload_collection(
        client,
        collection,
        CHARACTER_PAYLOAD_INDEXES,
        recreate=recreate,
    )


async def ensure_user_collection(
    client: AsyncQdrantClient,
    collection: str,
    *,
    recreate: bool = False,
) -> None:
    """Создать коллекцию учётных записей.

    Тот же приём, что и с образами: строк мало, искать по ним нужно ровно по
    адресу почты, поэтому коллекция нужна как хранилище, а не как индекс.
    """
    await _ensure_payload_collection(
        client,
        collection,
        USER_PAYLOAD_INDEXES,
        recreate=recreate,
    )


async def ensure_refresh_session_collection(
    client: AsyncQdrantClient,
    collection: str,
    *,
    recreate: bool = False,
) -> None:
    """Создать коллекцию выданных refresh-сессий.

    Записи здесь не удаляются по истечении срока: погашенная сессия — это звено
    цепочки ротаций, по которому видно, что токен предъявили повторно.
    """
    await _ensure_payload_collection(
        client,
        collection,
        REFRESH_SESSION_PAYLOAD_INDEXES,
        recreate=recreate,
    )


async def _ensure_payload_collection(
    client: AsyncQdrantClient,
    collection: str,
    indexes: Sequence[tuple[str, models.PayloadSchemaType]],
    *,
    recreate: bool,
) -> None:
    """Создать коллекцию без векторов и навесить её payload-индексы."""
    if recreate:
        await client.delete_collection(collection)

    if not await client.collection_exists(collection):
        await client.create_collection(collection_name=collection, vectors_config={})

    for field_name, field_schema in indexes:
        await client.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=field_schema,
        )
