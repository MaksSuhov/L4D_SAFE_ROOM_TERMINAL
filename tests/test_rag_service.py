"""Unit tests for the business-logic layer, running entirely on fakes."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.exceptions import AIServiceError, CharacterNotFoundError
from app.interfaces import RetrievedChunk
from app.services import (
    ArchivePromptBuilder,
    OpenAIClient,
    RAGService,
    RetrievalPolicy,
    TierPolicy,
)
from app.services.prompt_builder import NO_OFFICIAL_LORE
from app.taxonomy import LoreTopic, SourceType
from tests.fakes import (
    FakeAIClient,
    InMemoryCharacterRepository,
    InMemoryLoreChunkRepository,
    build_character,
    build_chunk,
)

QUESTION = "How do we kill a Tank?"
ANSWER = "Walls behind you, kids."


def block(context: str, header: str) -> str:
    """Return the body of one ``<header>…</header>`` block of the brief."""
    start = context.index(f"<{header}>")
    end = context.index(f"</{header}>")
    return context[start:end]


def tiered_matches() -> list[RetrievedChunk]:
    """Matches of every tier, ordered by descending similarity as a search returns them."""
    return [
        RetrievedChunk(
            chunk=build_chunk(
                chunk_id=10,
                content="Tank has 4000 HP.",
                source_type=SourceType.OFFICIAL,
                title="Tank › Overview",
            ),
            similarity=0.91,
        ),
        RetrievedChunk(
            chunk=build_chunk(
                chunk_id=11,
                content="Fight it with fire.",
                source_type=SourceType.OFFICIAL,
                title="Tank › Tactics",
            ),
            similarity=0.62,
        ),
        RetrievedChunk(
            chunk=build_chunk(
                chunk_id=12,
                content="Witch startle rules.",
                source_type=SourceType.OFFICIAL,
                title="Witch › Overview",
            ),
            similarity=0.31,
        ),
        RetrievedChunk(
            chunk=build_chunk(
                chunk_id=20,
                content="THE GOVERNMENT IS SPRAYING US",
                source_type=SourceType.GRAFFITI,
                topic=LoreTopic.GRAFFITI,
                title="No Mercy › The Subway",
            ),
            similarity=0.70,
        ),
        RetrievedChunk(
            chunk=build_chunk(
                chunk_id=21,
                content="NO ZOMBIE IS SAFE FROM CHICAGO TED",
                source_type=SourceType.GRAFFITI,
                topic=LoreTopic.GRAFFITI,
                title="No Mercy › Rooftop Finale",
            ),
            similarity=0.40,
        ),
        RetrievedChunk(
            chunk=build_chunk(
                chunk_id=30,
                content="We are a right damn good team!",
                source_type=SourceType.VOICE,
                topic=LoreTopic.QUOTES,
                title="Эллис · General",
                character_name="Эллис",
            ),
            similarity=0.55,
        ),
        RetrievedChunk(
            chunk=build_chunk(
                chunk_id=31,
                content="I believe in my pistol.",
                source_type=SourceType.VOICE,
                topic=LoreTopic.QUOTES,
                title="Ник · General",
                character_name="Ник",
            ),
            similarity=0.99,
        ),
    ]


def policy(
    *,
    official: TierPolicy | None = None,
    graffiti: TierPolicy | None = None,
    voice: TierPolicy | None = None,
) -> RetrievalPolicy:
    """Build a policy, defaulting to the shipped one for the tiers left out."""
    defaults = RetrievalPolicy()
    return RetrievalPolicy(
        official=official or defaults.official,
        graffiti=graffiti or defaults.graffiti,
        voice=voice or defaults.voice,
    )


def build_service(
    *,
    similarity: float = 0.0,
    policy_override: RetrievalPolicy | None = None,
    examples: list[dict[str, str]] | None = None,
) -> tuple[RAGService, InMemoryLoreChunkRepository, FakeAIClient]:
    character = build_character(few_shot_examples=examples)
    lore = InMemoryLoreChunkRepository(tiered_matches())
    ai_client = FakeAIClient(answer=ANSWER)
    service = RAGService(
        InMemoryCharacterRepository([character]),
        lore,
        ArchivePromptBuilder(),
        ai_client,
        policy=policy_override
        or policy(
            official=TierPolicy(limit=4, min_similarity=similarity),
            graffiti=TierPolicy(limit=3, min_similarity=similarity),
            voice=TierPolicy(limit=2, min_similarity=similarity),
        ),
    )
    return service, lore, ai_client


async def test_process_chat_message_returns_answer_with_sources() -> None:
    service, _, ai_client = build_service(
        examples=[{"user": "hi", "assistant": "hey there"}],
    )

    response = await service.process_chat_message(1, QUESTION)

    assert response.character_id == 1
    assert response.character_name == "Эллис"
    assert response.answer == ANSWER
    assert response.model == "fake-model"
    assert ai_client.embedded_texts == [QUESTION]


async def test_sources_carry_the_tier_of_every_record() -> None:
    service, _, _ = build_service()

    response = await service.process_chat_message(1, QUESTION)

    assert [(source.id, source.source_type) for source in response.sources] == [
        (10, SourceType.OFFICIAL),
        (11, SourceType.OFFICIAL),
        (12, SourceType.OFFICIAL),
        (20, SourceType.GRAFFITI),
        (21, SourceType.GRAFFITI),
        (30, SourceType.VOICE),
    ]
    assert response.sources[3].title == "No Mercy › The Subway"
    assert response.sources[5].character_name == "Эллис"


async def test_every_tier_is_queried_with_its_own_budget() -> None:
    service, lore, _ = build_service()

    await service.process_chat_message(1, QUESTION)

    assert [
        (search["source_types"], search["limit"], search["character_name"])
        for search in lore.searches
    ] == [
        ([SourceType.OFFICIAL], 4, None),
        ([SourceType.GRAFFITI], 3, None),
        ([SourceType.VOICE], 2, "Эллис"),
    ]


async def test_facts_and_rumours_are_rendered_in_separate_blocks() -> None:
    service, _, ai_client = build_service()

    await service.process_chat_message(1, QUESTION)

    context = str(ai_client.generation_calls[0]["context"])
    official = block(context, "official_lore")
    graffiti = block(context, "wall_graffiti")
    voice = block(context, "speech_samples")

    assert "1. [Tank › Overview] Tank has 4000 HP." in official
    assert "1. [No Mercy › The Subway] THE GOVERNMENT IS SPRAYING US" in graffiti
    assert "— We are a right damn good team!" in voice


async def test_graffiti_never_leaks_into_the_official_block() -> None:
    """The whole point of the split: hearsay must not be presented as lore."""
    service, _, ai_client = build_service()

    await service.process_chat_message(1, QUESTION)

    context = str(ai_client.generation_calls[0]["context"])
    assert "THE GOVERNMENT IS SPRAYING US" not in block(context, "official_lore")
    assert "Tank has 4000 HP." not in block(context, "wall_graffiti")


async def test_voice_samples_stay_with_the_answering_character() -> None:
    service, _, ai_client = build_service()

    await service.process_chat_message(1, QUESTION)

    context = str(ai_client.generation_calls[0]["context"])
    assert "I believe in my pistol." not in context
    assert "We are a right damn good team!" in context


async def test_absent_tier_is_announced_instead_of_left_empty() -> None:
    service, _, ai_client = build_service(
        policy_override=policy(
            official=TierPolicy(limit=1),
            graffiti=TierPolicy(limit=0),
            voice=TierPolicy(limit=0),
        )
    )

    await service.process_chat_message(1, QUESTION)

    context = str(ai_client.generation_calls[0]["context"])
    assert NO_OFFICIAL_LORE not in block(context, "official_lore")
    assert "подходящих надписей на стенах не найдено" in block(context, "wall_graffiti")


async def test_a_wholly_empty_archive_produces_no_brief_at_all() -> None:
    """With nothing retrieved there is no brief; the provider then says so itself."""
    service, _, ai_client = build_service(
        policy_override=policy(
            official=TierPolicy(limit=0),
            graffiti=TierPolicy(limit=0),
            voice=TierPolicy(limit=0),
        )
    )

    await service.process_chat_message(1, QUESTION)

    assert ai_client.generation_calls[0]["context"] == ""


async def test_zero_budget_skips_the_tier_entirely() -> None:
    service, lore, _ = build_service(
        policy_override=policy(
            official=TierPolicy(limit=1),
            graffiti=TierPolicy(limit=0),
            voice=TierPolicy(limit=0),
        )
    )

    await service.process_chat_message(1, QUESTION)

    assert len(lore.searches) == 1
    assert lore.searches[0]["source_types"] == [SourceType.OFFICIAL]


async def test_the_voice_tier_is_ranked_but_not_gated_by_the_evidence_floor() -> None:
    """A character's own lines are a style reference, so they survive a strict floor."""
    service, _, ai_client = build_service(
        policy_override=policy(
            official=TierPolicy(limit=4, min_similarity=0.99),
            graffiti=TierPolicy(limit=3, min_similarity=0.99),
            voice=TierPolicy(limit=1, min_similarity=0.0),
        )
    )

    response = await service.process_chat_message(1, QUESTION)

    assert [source.id for source in response.sources] == [30]
    context = str(ai_client.generation_calls[0]["context"])
    assert "We are a right damn good team!" in context
    assert "Tank has 4000 HP." not in context


async def test_a_disabled_voice_tier_leaves_its_block_empty() -> None:
    service, _, ai_client = build_service(
        policy_override=policy(voice=TierPolicy(limit=0)),
    )

    await service.process_chat_message(1, QUESTION)

    context = str(ai_client.generation_calls[0]["context"])
    assert "<speech_samples>" in context
    assert "We are a right damn good team!" not in context


async def test_min_similarity_filters_each_tier() -> None:
    service, _, ai_client = build_service(similarity=0.65)

    response = await service.process_chat_message(1, QUESTION)

    assert [source.id for source in response.sources] == [10, 20]
    context = str(ai_client.generation_calls[0]["context"])
    assert "Witch startle rules." not in context
    assert "NO ZOMBIE IS SAFE FROM CHICAGO TED" not in context


async def test_prompt_contains_persona_context_and_examples() -> None:
    examples = [{"user": "hi", "assistant": "hey there"}]
    service, _, ai_client = build_service(
        policy_override=policy(
            official=TierPolicy(limit=2),
            graffiti=TierPolicy(limit=0),
            voice=TierPolicy(limit=0),
        ),
        examples=examples,
    )

    await service.process_chat_message(1, QUESTION)

    call = ai_client.generation_calls[0]
    assert call["system_prompt"] == "Ты — Эллис."
    assert call["user_message"] == QUESTION
    assert call["few_shot_examples"] == examples
    context = str(call["context"])
    assert "1. [Tank › Overview] Tank has 4000 HP." in context
    assert "2. [Tank › Tactics] Fight it with fire." in context
    assert "3. [Witch › Overview]" not in context


async def test_empty_archive_produces_empty_context() -> None:
    ai_client = FakeAIClient(answer=ANSWER)
    service = RAGService(
        InMemoryCharacterRepository([build_character()]),
        InMemoryLoreChunkRepository(),
        ArchivePromptBuilder(),
        ai_client,
    )

    response = await service.process_chat_message(1, QUESTION)

    assert response.sources == []
    assert ai_client.generation_calls[0]["context"] == ""


async def test_unknown_character_fails_before_calling_the_provider() -> None:
    ai_client = FakeAIClient()
    service = RAGService(
        InMemoryCharacterRepository(),
        InMemoryLoreChunkRepository(tiered_matches()),
        ArchivePromptBuilder(),
        ai_client,
    )

    with pytest.raises(CharacterNotFoundError):
        await service.process_chat_message(404, QUESTION)

    assert ai_client.embedded_texts == []
    assert ai_client.generation_calls == []


def test_openai_client_requires_an_api_key() -> None:
    with pytest.raises(AIServiceError):
        OpenAIClient(Settings(ai_api_key=""))
