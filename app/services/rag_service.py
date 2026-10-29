"""Retrieval-augmented generation pipeline.

The service orchestrates repositories and the AI client, receiving every
collaborator through the constructor (constructor injection), which makes the
pipeline testable without a database or a network connection.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.exceptions import CharacterNotFoundError
from app.interfaces import (
    AIClientProtocol,
    CharacterRepositoryProtocol,
    LoreChunkRepositoryProtocol,
    RetrievedChunk,
)
from app.schemas import ChatResponse, LoreSource


class RAGService:
    """Retrieval-augmented generation pipeline for one chat message.

    Flow: embed the question -> retrieve the closest lore chunks -> load the
    character persona -> assemble the prompt -> generate the answer.
    """

    def __init__(
        self,
        character_repository: CharacterRepositoryProtocol,
        lore_chunk_repository: LoreChunkRepositoryProtocol,
        ai_client: AIClientProtocol,
        *,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> None:
        self._characters = character_repository
        self._lore_chunks = lore_chunk_repository
        self._ai_client = ai_client
        self._top_k = top_k
        self._min_similarity = min_similarity

    async def process_chat_message(self, character_id: int, user_message: str) -> ChatResponse:
        """Answer ``user_message`` as the character identified by ``character_id``."""
        character = await self._characters.get_by_id(character_id)
        if character is None:
            raise CharacterNotFoundError(character_id)

        query_embedding = await self._ai_client.get_embedding(user_message)
        matches = await self._lore_chunks.search_similar_chunks(
            query_embedding,
            limit=self._top_k,
        )
        relevant_matches = self._filter_by_similarity(matches)

        answer = await self._ai_client.generate_response(
            system_prompt=character.system_prompt,
            context=self._build_context(relevant_matches),
            user_message=user_message,
            few_shot_examples=character.few_shot_examples,
        )

        return ChatResponse(
            character_id=character.id,
            character_name=character.name,
            answer=answer,
            model=self._ai_client.model_name,
            sources=self._build_sources(relevant_matches),
        )

    def _filter_by_similarity(
        self,
        matches: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Drop weak matches so unrelated questions do not pollute the prompt."""
        return [match for match in matches if match.similarity >= self._min_similarity]

    @staticmethod
    def _build_context(matches: Sequence[RetrievedChunk]) -> str:
        """Render retrieved chunks as a numbered list of archive records."""
        return "\n".join(
            f"{position}. [{match.chunk.data_type}] {match.chunk.content}"
            for position, match in enumerate(matches, start=1)
        )

    @staticmethod
    def _build_sources(matches: Sequence[RetrievedChunk]) -> list[LoreSource]:
        """Expose the evidence used for the answer to the API client."""
        return [
            LoreSource(
                id=match.chunk.id,
                data_type=match.chunk.data_type,
                content=match.chunk.content,
                similarity=round(match.similarity, 4),
            )
            for match in matches
        ]
