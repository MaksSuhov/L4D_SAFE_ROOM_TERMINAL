"""Репозиторий архива лора на Qdrant: гибридный поиск с косинусной близостью.

Порт требует, чтобы ``similarity`` оставалась косинусной близостью: на ней
держатся пороги уровней доказательств, и подмена её рангом слияния лишила бы
пороги смысла. Поэтому роли разделены:

* **Слияние решает, кто попадает в бюджет уровня.** Оба плеча отдают свои
  кандидаты, они сливаются по рангам, и бюджет заполняется в порядке слияния.
  Запись, которую лексическое плечо поставило первой, вытесняет ту, которую
  плотное лишь поставило второй, — в этом и состоит польза гибрида.
* **Косинус сообщается, а не отбирает.** Близость каждой находки считается по её
  плотному вектору и возвращается вызывающему, который по ней фильтрует.

Пересортировка по косинусу *до* отсечения бюджета обнулила бы гибрид: плотное
плечо уже отдаёт свой глобальный топ, поэтому верхушка объединения по косинусу
совпала бы с верхушкой одного плотного плеча.

Отсюда следует и честная граница применимости: при боевом пороге
``rag_min_similarity`` лексическое плечо работает внутри полосы записей, которые
этот порог прошли, и не вправе протащить запись со слабой близостью. Порог — это
решение домена («не кормить модель слабыми совпадениями»), а не свойство
хранилища, поэтому перебивать его здесь нельзя.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qdrant_client import AsyncQdrantClient, models

from app.domain import LoreChunk
from app.exceptions import VectorStoreError
from app.interfaces import RetrievedChunk
from app.schemas import EmbeddedLoreChunk
from app.taxonomy import SourceType
from app.vector.bm25 import SparseEncoderProtocol
from app.vector.schema import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

HASH_LOOKUP_CHUNK_SIZE: int = 256
"""Верхняя граница числа хешей в одном фильтре: большой корпус никогда не породит
запрос, который сервер откажется планировать.
"""

_SCORE_EPSILON: float = 1e-12
"""Защита от деления на ноль при нормировке вектора нулевой длины."""


class QdrantLoreChunkRepository:
    """Хранение и гибридный поиск по архиву лора в Qdrant."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        sparse_encoder: SparseEncoderProtocol,
        *,
        hybrid_enabled: bool = True,
        prefetch_multiplier: int = 4,
    ) -> None:
        self._client = client
        self._collection = collection
        self._sparse = sparse_encoder
        self._hybrid = hybrid_enabled
        self._prefetch_multiplier = prefetch_multiplier

    async def create_chunk(self, record: EmbeddedLoreChunk) -> LoreChunk:
        """Сохранить один проиндексированный фрагмент лора."""
        (chunk,) = await self.create_chunks([record])
        return chunk

    async def create_chunks(self, records: Sequence[EmbeddedLoreChunk]) -> Sequence[LoreChunk]:
        """Записать пакет фрагментов лора одной операцией.

        Идентификатор точки выводится из хеша текста, поэтому повторная запись
        той же записи перезаписывает её точку, а не создаёт вторую. Это делает
        наполнение идемпотентным без отдельного учёта уже сохранённого.
        """
        if not records:
            return []

        sparse_vectors = await self._sparse.encode_documents([record.content for record in records])
        points = [
            self._to_point(record, sparse_vector)
            for record, sparse_vector in zip(records, sparse_vectors, strict=True)
        ]
        await self._client.upsert(self._collection, points=points, wait=True)
        return [self._to_chunk(point.payload or {}) for point in points]

    async def search_similar_chunks(
        self,
        query_embedding: Sequence[float],
        query_text: str,
        *,
        limit: int,
        source_types: Sequence[SourceType] | None = None,
        topic: str | None = None,
        character_name: str | None = None,
    ) -> Sequence[RetrievedChunk]:
        """Найти один уровень доказательств, ведя поиск косинусом и BM25 сразу."""
        if limit <= 0:
            return []

        query_filter = self._build_filter(source_types, topic, character_name)
        if not self._hybrid:
            return await self._dense_only(query_embedding, query_filter, limit)

        candidates = await self._query_hybrid(query_embedding, query_text, query_filter, limit)
        matches = self._with_similarity(query_embedding, candidates[:limit])
        matches.sort(key=lambda match: match.similarity, reverse=True)
        return matches

    async def existing_content_hashes(self, hashes: Sequence[str]) -> set[str]:
        """Вернуть, какие из данных хешей содержимого уже сохранены."""
        wanted = list(dict.fromkeys(hashes))
        found: set[str] = set()
        for start in range(0, len(wanted), HASH_LOOKUP_CHUNK_SIZE):
            window = wanted[start : start + HASH_LOOKUP_CHUNK_SIZE]
            records, _ = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="content_hash",
                            match=models.MatchAny(any=window),
                        )
                    ]
                ),
                limit=len(window),
                with_payload=["content_hash"],
                with_vectors=False,
            )
            found.update(
                str(record.payload["content_hash"]) for record in records if record.payload
            )
        return found

    async def count(self) -> int:
        """Вернуть число сохранённых фрагментов лора."""
        result = await self._client.count(self._collection, exact=True)
        return int(result.count)

    async def delete_all(self) -> int:
        """Удалить все записи архива и сообщить, сколько их было.

        Метод намеренно отсутствует в порту репозитория: ни один сервис не вправе
        опустошить архив, только инструмент наполнения, владеющий его жизненным
        циклом.
        """
        removed = await self.count()
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[])),
            wait=True,
        )
        return removed

    @staticmethod
    def point_id(content_hash: str) -> int:
        """Вывести идентификатор точки из хеша содержимого.

        Шестнадцати шестнадцатеричных знаков хватает на 64 бита, а это ровно та
        ширина, которую Qdrant принимает как числовой идентификатор точки.
        """
        return int(content_hash[:16], 16)

    async def _query_hybrid(
        self,
        query_embedding: Sequence[float],
        query_text: str,
        query_filter: models.Filter | None,
        limit: int,
    ) -> Sequence[models.ScoredPoint]:
        """Спросить оба плеча сразу и слить их выдачу по рангам.

        Каждое плечо видит один и тот же фильтр уровня: гибрид расширяет воронку
        внутри уровня доказательств, а не поверх его границы. Воронка шире бюджета
        намеренно — у плеч должно быть место разойтись, иначе лексическое плечо не
        может принести ничего нового.
        """
        window = limit * self._prefetch_multiplier
        response = await self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                models.Prefetch(
                    query=list(query_embedding),
                    using=DENSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=window,
                ),
                models.Prefetch(
                    query=await self._sparse.encode_query(query_text),
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=window,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=window,
            with_payload=True,
            with_vectors=[DENSE_VECTOR_NAME],
        )
        return response.points

    async def _dense_only(
        self,
        query_embedding: Sequence[float],
        query_filter: models.Filter | None,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Поиск только по косинусу: выключатель гибрида и аварийный путь."""
        response = await self._client.query_points(
            collection_name=self._collection,
            query=list(query_embedding),
            using=DENSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return [
            RetrievedChunk(chunk=self._to_chunk(point.payload or {}), similarity=float(point.score))
            for point in response.points
        ]

    def _with_similarity(
        self,
        query_embedding: Sequence[float],
        candidates: Sequence[models.ScoredPoint],
    ) -> list[RetrievedChunk]:
        """Посчитать косинусную близость кандидатов, сохранив порядок слияния.

        Эмбеддинги провайдера приходят нормированными, но нормировка повторяется:
        косинус обязан остаться косинусом, чем бы плотность вектора ни задавалась.
        """
        query = np.asarray(query_embedding, dtype=np.float32)
        query /= np.linalg.norm(query) + _SCORE_EPSILON

        matches: list[RetrievedChunk] = []
        for point in candidates:
            vector = self._dense_vector(point)
            if vector is None:
                continue
            vector /= np.linalg.norm(vector) + _SCORE_EPSILON
            matches.append(
                RetrievedChunk(
                    chunk=self._to_chunk(point.payload or {}),
                    similarity=float(np.dot(query, vector)),
                )
            )

        if candidates and not matches:
            # The store answered with points but not with their vectors, so every
            # similarity would silently collapse to "no match". Failing loudly beats
            # returning an empty archive to a question that has an answer.
            raise VectorStoreError(
                "Qdrant returned no stored dense vectors; the collection was created "
                "without the named vector the repository searches"
            )

        return matches

    @staticmethod
    def _dense_vector(point: models.ScoredPoint) -> np.ndarray | None:
        """Достать плотный вектор из ответа Qdrant, если он вернулся."""
        vectors = point.vector
        if not isinstance(vectors, dict):
            return None
        raw = vectors.get(DENSE_VECTOR_NAME)
        if raw is None:
            return None
        return np.asarray(raw, dtype=np.float32)

    def _to_point(
        self,
        record: EmbeddedLoreChunk,
        sparse_vector: models.SparseVector,
    ) -> models.PointStruct:
        """Собрать точку Qdrant из записи корпуса и её разреженного вектора."""
        identifier = self.point_id(record.content_hash)
        return models.PointStruct(
            id=identifier,
            vector={
                DENSE_VECTOR_NAME: list(record.embedding),
                SPARSE_VECTOR_NAME: sparse_vector,
            },
            payload={
                "id": identifier,
                "content": record.content,
                "content_hash": record.content_hash,
                "source_type": str(record.source_type),
                "topic": str(record.topic),
                "title": record.title,
                "character_name": record.character_name,
                "source_url": record.source_url,
            },
        )

    @staticmethod
    def _to_chunk(payload: dict[str, object]) -> LoreChunk:
        """Собрать доменную сущность из payload точки.

        Значения payload приходят без типа, поэтому идентификатор приводится через
        строку: так одно и то же выражение принимает и число, и его текстовую форму,
        которую Qdrant вправе вернуть.
        """
        character_name = payload.get("character_name")
        return LoreChunk(
            content=str(payload["content"]),
            content_hash=str(payload["content_hash"]),
            source_type=str(payload["source_type"]),
            topic=str(payload["topic"]),
            title=str(payload["title"]),
            id=int(str(payload["id"])),
            character_name=str(character_name) if character_name is not None else None,
            source_url=str(payload.get("source_url") or ""),
        )

    @staticmethod
    def _build_filter(
        source_types: Sequence[SourceType] | None,
        topic: str | None,
        character_name: str | None,
    ) -> models.Filter | None:
        """Собрать фильтр уровня доказательств из тех же условий, что и раньше в SQL."""
        must: list[models.Condition] = []
        if source_types is not None:
            must.append(
                models.FieldCondition(
                    key="source_type",
                    match=models.MatchAny(any=[str(source_type) for source_type in source_types]),
                )
            )
        if topic is not None:
            must.append(models.FieldCondition(key="topic", match=models.MatchValue(value=topic)))
        if character_name is not None:
            must.append(
                models.FieldCondition(
                    key="character_name",
                    match=models.MatchValue(value=character_name),
                )
            )
        return models.Filter(must=must) if must else None
