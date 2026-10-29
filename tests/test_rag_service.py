"""Unit tests for the business-logic layer, running entirely on fakes."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.exceptions import AIServiceError, CharacterNotFoundError
from app.interfaces import RetrievedChunk
from app.schemas import LoreChunkCreate
from app.services import LoreIngestionService, OpenAIClient, RAGService
from tests.fakes import (
    FakeAIClient,
    InMemoryCharacterRepository,
    InMemoryLoreChunkRepository,
    build_character,
    build_chunk,
)

QUESTION = "How do we kill a Tank?"
ANSWER = "Walls behind you, kids."


def three_matches() -> list[RetrievedChunk]:
    """Matches already ordered by descending similarity, as the repository returns them."""
    return [
        RetrievedChunk(build_chunk(chunk_id=10, content="Tank has 4000 HP."), 0.91),
        RetrievedChunk(build_chunk(chunk_id=11, content="Fight it with fire."), 0.62),
        RetrievedChunk(build_chunk(chunk_id=12, content="Witch startle rules."), 0.31),
    ]


def build_service(
    *,
    similarity: float = 0.0,
    top_k: int = 3,
    examples: list[dict[str, str]] | None = None,
) -> tuple[RAGService, InMemoryLoreChunkRepository, FakeAIClient]:
    character = build_character(few_shot_examples=examples)
    lore = InMemoryLoreChunkRepository(three_matches())
    ai_client = FakeAIClient(answer=ANSWER)
    service = RAGService(
        InMemoryCharacterRepository([character]),
        lore,
        ai_client,
        top_k=top_k,
        min_similarity=similarity,
    )
    return service, lore, ai_client


async def test_process_chat_message_returns_answer_with_sources() -> None:
    service, lore, ai_client = build_service(
        examples=[{"user": "hi", "assistant": "hey there"}],
    )

    response = await service.process_chat_message(1, QUESTION)

    assert response.character_id == 1
    assert response.character_name == "Ellis"
    assert response.answer == ANSWER
    assert response.model == "fake-model"
    assert [source.id for source in response.sources] == [10, 11, 12]
    assert response.sources[0].similarity == 0.91
    assert ai_client.embedded_texts == [QUESTION]
    assert lore.searches[0]["limit"] == 3


async def test_prompt_contains_persona_context_and_examples() -> None:
    examples = [{"user": "hi", "assistant": "hey there"}]
    service, _, ai_client = build_service(top_k=2, examples=examples)

    await service.process_chat_message(1, QUESTION)

    call = ai_client.generation_calls[0]
    assert call["system_prompt"] == "You are Ellis."
    assert call["user_message"] == QUESTION
    assert call["few_shot_examples"] == examples
    context = str(call["context"])
    assert context.startswith("1. [monster] Tank has 4000 HP.")
    assert "2. [monster] Fight it with fire." in context
    assert "3. [monster]" not in context


async def test_min_similarity_filters_weak_matches() -> None:
    service, _, ai_client = build_service(similarity=0.6)

    response = await service.process_chat_message(1, QUESTION)

    assert [source.id for source in response.sources] == [10, 11]
    assert "3. [monster]" not in str(ai_client.generation_calls[0]["context"])


async def test_empty_archive_produces_empty_context() -> None:
    ai_client = FakeAIClient(answer=ANSWER)
    service = RAGService(
        InMemoryCharacterRepository([build_character()]),
        InMemoryLoreChunkRepository(),
        ai_client,
    )

    response = await service.process_chat_message(1, QUESTION)

    assert response.sources == []
    assert ai_client.generation_calls[0]["context"] == ""


async def test_unknown_character_fails_before_calling_the_provider() -> None:
    ai_client = FakeAIClient()
    service = RAGService(
        InMemoryCharacterRepository(),
        InMemoryLoreChunkRepository(three_matches()),
        ai_client,
    )

    with pytest.raises(CharacterNotFoundError):
        await service.process_chat_message(404, QUESTION)

    assert ai_client.embedded_texts == []
    assert ai_client.generation_calls == []


async def test_ingestion_service_embeds_and_stores_chunks() -> None:
    repository = InMemoryLoreChunkRepository()
    ai_client = FakeAIClient()
    service = LoreIngestionService(repository, ai_client)

    stored = await service.ingest(
        [
            LoreChunkCreate(content="Tank lore", data_type="monster"),
            LoreChunkCreate(content="Witch lore", data_type="monster"),
        ]
    )

    assert stored == 2
    assert ai_client.embedded_texts == ["Tank lore", "Witch lore"]
    assert [chunk.content for chunk in repository.stored] == ["Tank lore", "Witch lore"]


async def test_ingestion_without_items_does_not_call_the_provider() -> None:
    ai_client = FakeAIClient()
    service = LoreIngestionService(InMemoryLoreChunkRepository(), ai_client)

    assert await service.ingest([]) == 0
    assert ai_client.embedded_texts == []


def test_openai_client_requires_an_api_key() -> None:
    with pytest.raises(AIServiceError):
        OpenAIClient(Settings(openai_api_key=""))
