from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.exceptions import CharacterNotFoundError
from app.interfaces import (
    AIClientProtocol,
    CharacterRepositoryProtocol,
    LoreChunkRepositoryProtocol,
    PreparedChat,
    QueryExpanderProtocol,
    RetrievedChunk,
    TracerProtocol,
)
from app.schemas import ChatResponse, LoreSource
from app.services.prompt_builder import ArchiveEvidence, ArchivePromptBuilder
from app.services.query_expander import NoQueryExpander
from app.services.tracing import NoOpTracer
from app.taxonomy import SourceType

logger = logging.getLogger("l4d.rag")


@dataclass(frozen=True, slots=True)
class RetrievalProbe:
    """Одна поисковая формулировка вместе с эмбеддингом своего текста."""

    query: str
    embedding: Sequence[float]


@dataclass(frozen=True, slots=True)
class TierPolicy:
    """Сколько один уровень доказательств может дать ответу и насколько близкими
    должны быть совпадения.
    """

    limit: int = 0
    min_similarity: float = 0.0

    @property
    def is_enabled(self) -> bool:
        """Правда ли, что уровень вообще запрашивается."""
        return self.limit > 0


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    """Бюджет и порог косинусной близости для каждого уровня.

    Пороги задаются отдельно по уровням, потому что уровни служат разным целям.
    Проверенный лор и настенные надписи — это *доказательства*: слабое совпадение
    это шум, который испортил бы ответ, поэтому совпадение должно заслужить своё
    место. Собственные реплики персонажа — *образец стиля*: они уже сужены до
    одного персонажа, и две самые релевантные из них полезны даже тогда, когда
    вопрос совсем не о том, для чего эти реплики записывались.
    """

    official: TierPolicy = TierPolicy(limit=4, min_similarity=0.25)
    graffiti: TierPolicy = TierPolicy(limit=3, min_similarity=0.25)
    voice: TierPolicy = TierPolicy(limit=2, min_similarity=0.0)

    def for_tier(self, source_type: SourceType) -> TierPolicy:
        """Вернуть политику для ``source_type``."""
        return {
            SourceType.OFFICIAL: self.official,
            SourceType.GRAFFITI: self.graffiti,
            SourceType.VOICE: self.voice,
        }[source_type]


class RAGService:
    """Двухуровневая генерация с дополнением из поиска для одного сообщения чата.

    Поток: развернуть вопрос в поисковые формулировки (HIDE) -> получить их
    эмбеддинги -> найти три уровня доказательств по отдельности (проверенный лор,
    настенные надписи, собственные реплики персонажа) -> загрузить образ -> собрать
    архивную сводку -> сгенерировать ответ.

    Уровни находятся отдельными запросами со своими бюджетами. В этом суть
    конвейера: стена слухов не может вытеснить проверенный лор из контекста,
    потому что они никогда не конкурируют за одни и те же места. Формулировок же
    может быть несколько — они лишь решают, *какие* записи займут эти места.
    """

    def __init__(
        self,
        character_repository: CharacterRepositoryProtocol,
        lore_chunk_repository: LoreChunkRepositoryProtocol,
        prompt_builder: ArchivePromptBuilder,
        ai_client: AIClientProtocol,
        *,
        policy: RetrievalPolicy | None = None,
        query_expander: QueryExpanderProtocol | None = None,
        tracer: TracerProtocol | None = None,
    ) -> None:
        self._characters = character_repository
        self._lore_chunks = lore_chunk_repository
        self._prompt_builder = prompt_builder
        self._ai_client = ai_client
        self._policy = policy or RetrievalPolicy()
        self._query_expander = query_expander or NoQueryExpander()
        self._tracer = tracer or NoOpTracer()

    async def process_chat_message(self, character_id: int, user_message: str) -> ChatResponse:
        """Ответить на ``user_message`` от лица персонажа с ``character_id``."""
        with self._tracer.span("chat") as span:
            prepared = await self.build_chat_context(character_id, user_message)

            answer = await self._ai_client.generate_response(
                system_prompt=prepared.system_prompt,
                context=prepared.context,
                user_message=user_message,
                few_shot_examples=prepared.few_shot_examples,
            )

            span.annotate(
                input={"character_id": character_id, "message": user_message},
                output=answer,
                metadata={"character_id": character_id},
            )

        return ChatResponse(
            character_id=prepared.character_id,
            character_name=prepared.character_name,
            answer=answer,
            model=prepared.model,
            sources=list(prepared.sources),
        )

    async def build_chat_context(self, character_id: int, user_message: str) -> PreparedChat:
        """Выполнить поиск для одного сообщения и вернуть всё, что нужно генерации.

        Половина конвейера, отвечающая за поиск, выставлена отдельно: HTTP-маршрут
        пользуется ею косвенно, а стенд оценки — напрямую, чтобы стримить ответ и
        замерять время до первого токена.
        """
        character = await self._characters.get_by_id(character_id)
        if character is None or character.id is None:
            # A persona that was never stored carries no identifier, so a caller
            # holding only an identifier cannot be talking about it.
            raise CharacterNotFoundError(character_id)
        identifier = character.id

        probes = await self._build_probes(user_message)
        evidence = await self._retrieve_evidence(probes, character.name)

        return PreparedChat(
            character_id=identifier,
            character_name=character.name,
            model=self._ai_client.model_name,
            system_prompt=character.system_prompt,
            context=self._prompt_builder.build(evidence),
            few_shot_examples=character.few_shot_examples,
            sources=self._build_sources(evidence.all_chunks),
        )

    async def _build_probes(self, question: str) -> list[RetrievalProbe]:
        """Развернуть вопрос в формулировки HIDE и посчитать их эмбеддинги.

        Все формулировки уходят одним пакетным запросом, поэтому расширение стоит
        одного обращения к провайдеру, сколько бы формулировок ни вернулось.
        """
        expanded = await self._query_expander.expand(question)
        vectors = await self._ai_client.get_embeddings(list(expanded.probes))
        probes = [
            RetrievalProbe(query=query, embedding=vector)
            for query, vector in zip(expanded.probes, vectors, strict=True)
        ]
        logger.debug("Retrieving for %d probes: %s", len(probes), [p.query for p in probes])
        return probes

    async def _retrieve_evidence(
        self,
        probes: Sequence[RetrievalProbe],
        character_name: str,
    ) -> ArchiveEvidence:
        """Найти каждый уровень отдельно, применяя его собственный бюджет и порог."""
        with self._tracer.span("retrieve", as_type="retriever") as span:
            evidence = ArchiveEvidence(
                official=await self._retrieve(probes, SourceType.OFFICIAL),
                graffiti=await self._retrieve(probes, SourceType.GRAFFITI),
                voice=await self._retrieve(
                    probes,
                    SourceType.VOICE,
                    character_name=character_name,
                ),
            )
            span.annotate(metadata=self._evidence_metadata(probes, evidence))

        return evidence

    @staticmethod
    def _evidence_metadata(
        probes: Sequence[RetrievalProbe],
        evidence: ArchiveEvidence,
    ) -> dict[str, object]:
        """Собрать находки для трейса: поисковые формулировки и пары «id, близость»."""
        return {
            "probes": [probe.query for probe in probes],
            "official": [
                (match.chunk.id, round(match.similarity, 3)) for match in evidence.official
            ],
            "graffiti": [
                (match.chunk.id, round(match.similarity, 3)) for match in evidence.graffiti
            ],
            "voice": [
                (match.chunk.id, round(match.similarity, 3)) for match in evidence.voice
            ],
        }

    async def _retrieve(
        self,
        probes: Sequence[RetrievalProbe],
        source_type: SourceType,
        *,
        character_name: str | None = None,
    ) -> list[RetrievedChunk]:
        """Найти один уровень и отбросить совпадения, слишком далёкие для доказательств.

        Каждая формулировка ищется отдельно, а найденное сливается по близости: запись
        занимает место в бюджете уровня, если её принесла хотя бы одна формулировка, и
        сохраняет близость той из них, для которой она оказалась ближе всего. Формулировки
        перебираются по порядку, поэтому при равной близости в бюджет попадает находка
        более ранней из них, а исходный вопрос идёт первым.
        """
        policy = self._policy.for_tier(source_type)
        if not policy.is_enabled:
            return []

        fused: dict[str, RetrievedChunk] = {}
        for probe in probes:
            matches = await self._lore_chunks.search_similar_chunks(
                probe.embedding,
                probe.query,
                limit=policy.limit,
                source_types=[source_type],
                character_name=character_name,
            )
            self._fuse(fused, matches, min_similarity=policy.min_similarity)

        ranked = sorted(fused.values(), key=lambda match: match.similarity, reverse=True)
        return ranked[: policy.limit]

    @staticmethod
    def _fuse(
        fused: dict[str, RetrievedChunk],
        matches: Sequence[RetrievedChunk],
        *,
        min_similarity: float,
    ) -> None:
        """Добавить выдачу одной формулировки, оставив у записи её лучшую близость.

        Ключ слияния — хэш содержимого: это и есть тождество записи в архиве, по нему
        же ингестинг решает, что уже загружено, так что одна и та же запись, найденная
        двумя формулировками, не занимает два места в бюджете уровня.
        """
        for match in matches:
            if match.similarity < min_similarity:
                continue
            best = fused.get(match.chunk.content_hash)
            if best is None or match.similarity > best.similarity:
                fused[match.chunk.content_hash] = match

    @staticmethod
    def _build_sources(matches: Sequence[RetrievedChunk]) -> list[LoreSource]:
        """Показать клиенту API доказательства, использованные в ответе."""
        return [
            LoreSource(
                id=match.chunk.id,
                source_type=match.chunk.source_type,
                topic=match.chunk.topic,
                title=match.chunk.title,
                character_name=match.chunk.character_name,
                content=match.chunk.content,
                similarity=round(match.similarity, 4),
            )
            for match in matches
        ]


__all__ = ["RAGService", "RetrievalPolicy", "RetrievalProbe", "TierPolicy"]
