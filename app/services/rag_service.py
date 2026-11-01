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
    """How much one evidence tier may contribute to an answer, and how close."""

    limit: int = 0
    min_similarity: float = 0.0

    @property
    def is_enabled(self) -> bool:
        """Whether the tier is consulted at all."""
        return self.limit > 0


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    """The budget and the similarity floor of every tier.

    The floors are per tier because the tiers serve different purposes. Verified
    lore and wall graffiti are *evidence*: a weak match is noise that would make
    the answer worse, so a match has to earn its place. The character's own lines
    are a *style reference*: they are already narrowed to one character, and the
    two most relevant of their lines are useful even when the question has
    nothing to do with what those lines were recorded for.
    """

    official: TierPolicy = TierPolicy(limit=4, min_similarity=0.25)
    graffiti: TierPolicy = TierPolicy(limit=3, min_similarity=0.25)
    voice: TierPolicy = TierPolicy(limit=2, min_similarity=0.0)

    def for_tier(self, source_type: SourceType) -> TierPolicy:
        """Return the policy of ``source_type``."""
        return {
            SourceType.OFFICIAL: self.official,
            SourceType.GRAFFITI: self.graffiti,
            SourceType.VOICE: self.voice,
        }[source_type]


class RAGService:
    """Two-level retrieval-augmented generation for one chat message.

    Flow: embed the question -> retrieve the three evidence tiers separately
    (verified lore, wall graffiti, the character's own lines) -> load the
    persona -> assemble the archive brief -> generate the answer.

    The tiers are retrieved in independent queries with their own budgets. That
    is the point of the pipeline: a wall of rumours cannot crowd the verified
    lore out of the context, because the two never compete for the same slots.
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
        """Answer ``user_message`` as the character identified by ``character_id``."""
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
        """Run retrieval for one message and return everything generation needs.

        The retrieval half of the pipeline, exposed on its own: the HTTP route
        uses it indirectly, while the evaluation harness uses it directly so it
        can stream the answer and time the first token.
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
        """Look every tier up separately, applying its own budget and floor."""
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
        """Search one tier and drop the matches that are too far to be evidence."""
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
        """Expose the evidence used for the answer to the API client."""
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
