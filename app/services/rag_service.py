from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.exceptions import CharacterNotFoundError
from app.interfaces import (
    AIClientProtocol,
    CharacterRepositoryProtocol,
    LoreChunkRepositoryProtocol,
    PreparedChat,
    RetrievedChunk,
)
from app.schemas import ChatResponse, LoreSource
from app.services.prompt_builder import ArchiveEvidence, ArchivePromptBuilder
from app.taxonomy import SourceType


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

    Поток: получить эмбеддинг вопроса -> найти три уровня доказательств по
    отдельности (проверенный лор, настенные надписи, собственные реплики
    персонажа) -> загрузить образ -> собрать архивную сводку -> сгенерировать
    ответ.

    Уровни находятся отдельными запросами со своими бюджетами. В этом суть
    конвейера: стена слухов не может вытеснить проверенный лор из контекста,
    потому что они никогда не конкурируют за одни и те же места.
    """

    def __init__(
        self,
        character_repository: CharacterRepositoryProtocol,
        lore_chunk_repository: LoreChunkRepositoryProtocol,
        prompt_builder: ArchivePromptBuilder,
        ai_client: AIClientProtocol,
        *,
        policy: RetrievalPolicy | None = None,
    ) -> None:
        self._characters = character_repository
        self._lore_chunks = lore_chunk_repository
        self._prompt_builder = prompt_builder
        self._ai_client = ai_client
        self._policy = policy or RetrievalPolicy()

    async def process_chat_message(self, character_id: int, user_message: str) -> ChatResponse:
        """Ответить на ``user_message`` от лица персонажа с ``character_id``."""
        prepared = await self.build_chat_context(character_id, user_message)

        answer = await self._ai_client.generate_response(
            system_prompt=prepared.system_prompt,
            context=prepared.context,
            user_message=user_message,
            few_shot_examples=prepared.few_shot_examples,
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
        if character is None:
            raise CharacterNotFoundError(character_id)

        query_embedding = await self._ai_client.get_embedding(user_message)
        evidence = await self._retrieve_evidence(query_embedding, character.name)

        return PreparedChat(
            character_id=character.id,
            character_name=character.name,
            model=self._ai_client.model_name,
            system_prompt=character.system_prompt,
            context=self._prompt_builder.build(evidence),
            few_shot_examples=character.few_shot_examples,
            sources=self._build_sources(evidence.all_chunks),
        )

    async def _retrieve_evidence(
        self,
        query_embedding: Sequence[float],
        character_name: str,
    ) -> ArchiveEvidence:
        """Найти каждый уровень отдельно, применяя его собственный бюджет и порог."""
        return ArchiveEvidence(
            official=await self._retrieve(query_embedding, SourceType.OFFICIAL),
            graffiti=await self._retrieve(query_embedding, SourceType.GRAFFITI),
            voice=await self._retrieve(
                query_embedding,
                SourceType.VOICE,
                character_name=character_name,
            ),
        )

    async def _retrieve(
        self,
        query_embedding: Sequence[float],
        source_type: SourceType,
        *,
        character_name: str | None = None,
    ) -> list[RetrievedChunk]:
        """Найти один уровень и отбросить совпадения, слишком далёкие для доказательств."""
        policy = self._policy.for_tier(source_type)
        if not policy.is_enabled:
            return []

        matches = await self._lore_chunks.search_similar_chunks(
            query_embedding,
            limit=policy.limit,
            source_types=[source_type],
            character_name=character_name,
        )
        return [match for match in matches if match.similarity >= policy.min_similarity]

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


__all__ = ["RAGService", "RetrievalPolicy", "TierPolicy"]
