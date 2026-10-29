"""Tests for the archive brief: the wording that separates facts from rumours."""

from __future__ import annotations

from app.interfaces import RetrievedChunk
from app.services import ArchiveEvidence, ArchivePromptBuilder
from app.services.prompt_builder import (
    GRAFFITI_SECTION_HEADER,
    NO_GRAFFITI,
    NO_OFFICIAL_LORE,
    OFFICIAL_SECTION_HEADER,
    VOICE_SECTION_HEADER,
)
from app.taxonomy import LoreTopic, SourceType
from tests.fakes import build_chunk


def match(chunk_id: int, content: str, **kwargs: object) -> RetrievedChunk:
    defaults: dict[str, object] = {
        "source_type": SourceType.OFFICIAL,
        "topic": LoreTopic.INFECTED,
        "title": "Tank › Overview",
    }
    defaults.update(kwargs)
    chunk = build_chunk(chunk_id=chunk_id, content=content, **defaults)  # type: ignore[arg-type]
    return RetrievedChunk(chunk=chunk, similarity=0.8)


def test_no_evidence_produces_no_brief() -> None:
    assert ArchivePromptBuilder().build(ArchiveEvidence()) == ""


def test_a_missing_tier_is_announced_rather_than_left_blank() -> None:
    brief = ArchivePromptBuilder().build(ArchiveEvidence(official=[match(1, "Tank has 4000 HP.")]))

    assert NO_GRAFFITI in brief
    assert NO_OFFICIAL_LORE not in brief


def test_every_record_is_numbered_and_located() -> None:
    brief = ArchivePromptBuilder().build(
        ArchiveEvidence(
            official=[
                match(1, "Tank has 4000 HP."),
                match(2, "Fire hurts it.", title="Tank › Tactics"),
            ],
            graffiti=[
                match(
                    3,
                    "SPRAYED BY THE GOVERNMENT",
                    source_type=SourceType.GRAFFITI,
                    topic=LoreTopic.GRAFFITI,
                    title="No Mercy › The Subway",
                )
            ],
        )
    )

    assert "1. [Tank › Overview] Tank has 4000 HP." in brief
    assert "2. [Tank › Tactics] Fire hurts it." in brief
    assert "1. [No Mercy › The Subway] SPRAYED BY THE GOVERNMENT" in brief


def test_the_brief_states_that_graffiti_is_hearsay() -> None:
    brief = ArchivePromptBuilder().build(
        ArchiveEvidence(official=[match(1, "fact", source_type=SourceType.OFFICIAL)])
    )

    assert "<wall_graffiti>" in brief
    assert "слухи" in brief
    assert "на стене" in brief


def test_voice_samples_become_one_bullet_per_line() -> None:
    brief = ArchivePromptBuilder().build(
        ArchiveEvidence(
            voice=[
                match(
                    7,
                    "We are a right damn good team!\nI am proud to be part of this team!",
                    source_type=SourceType.VOICE,
                    topic=LoreTopic.QUOTES,
                    title="Эллис · General",
                    character_name="Эллис",
                )
            ]
        )
    )

    assert "— We are a right damn good team!" in brief
    assert "— I am proud to be part of this team!" in brief


def test_the_three_blocks_appear_in_a_stable_order() -> None:
    brief = ArchivePromptBuilder().build(
        ArchiveEvidence(
            official=[match(1, "fact")],
            graffiti=[match(2, "rumour", source_type=SourceType.GRAFFITI)],
            voice=[match(3, "line", source_type=SourceType.VOICE)],
        )
    )

    assert (
        brief.index(OFFICIAL_SECTION_HEADER)
        < brief.index(GRAFFITI_SECTION_HEADER)
        < brief.index(VOICE_SECTION_HEADER)
    )


def test_all_chunks_lists_every_tier_for_the_api_response() -> None:
    evidence = ArchiveEvidence(
        official=[match(1, "fact")],
        graffiti=[match(2, "rumour", source_type=SourceType.GRAFFITI)],
        voice=[match(3, "line", source_type=SourceType.VOICE)],
    )

    assert [item.chunk.id for item in evidence.all_chunks] == [1, 2, 3]
    assert not evidence.is_empty
