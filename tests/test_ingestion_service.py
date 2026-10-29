"""Tests for the ingestion services: embedding, deduplication, snapshot, translation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.exceptions import ApplicationError
from app.ingestion import CorpusSnapshot, CorpusTranslator
from app.schemas import LoreChunkCreate
from app.services import LoreIngestionService, content_hash
from app.taxonomy import LoreTopic, SourceType
from tests.fakes import FakeAIClient, InMemoryLoreChunkRepository, build_document


def bullets() -> list[LoreChunkCreate]:
    return [
        build_document(
            content="NO CURE",
            source_type=SourceType.GRAFFITI,
            topic=LoreTopic.GRAFFITI,
        ),
        build_document(
            content="MERCY HOSPITAL IS OVERRUN",
            source_type=SourceType.GRAFFITI,
            topic=LoreTopic.GRAFFITI,
        ),
        build_document(content="Tank has 4000 HP.", topic=LoreTopic.INFECTED),
    ]


class TestContentHash:
    def test_ignores_whitespace_differences(self) -> None:
        assert content_hash("NO   CURE\n") == content_hash("NO CURE")

    def test_separates_different_texts(self) -> None:
        assert content_hash("NO CURE") != content_hash("NO CURE!")


class TestLoreIngestionService:
    async def test_embeds_and_stores_every_record(self) -> None:
        repository = InMemoryLoreChunkRepository()
        ai_client = FakeAIClient()

        report = await LoreIngestionService(repository, ai_client).ingest(bullets())

        assert report.stored == 3
        assert report.skipped == 0
        assert len(repository.stored) == 3

    async def test_records_keep_their_tier_and_provenance(self) -> None:
        repository = InMemoryLoreChunkRepository()

        await LoreIngestionService(repository, FakeAIClient()).ingest(bullets())

        stored = {chunk.content: chunk for chunk in repository.stored}
        assert stored["NO CURE"].source_type == SourceType.GRAFFITI
        assert stored["NO CURE"].topic == LoreTopic.GRAFFITI
        assert stored["Tank has 4000 HP."].source_type == SourceType.OFFICIAL

    async def test_reports_what_it_stored_per_tier(self) -> None:
        report = await LoreIngestionService(InMemoryLoreChunkRepository(), FakeAIClient()).ingest(
            bullets()
        )

        assert report.by_source_type == {"graffiti": 2, "official": 1}

    async def test_second_run_stores_nothing_and_calls_no_provider(self) -> None:
        repository = InMemoryLoreChunkRepository()
        ai_client = FakeAIClient()
        service = LoreIngestionService(repository, ai_client)
        await service.ingest(bullets())
        ai_client.embedded_texts.clear()

        report = await service.ingest(bullets())

        assert report.stored == 0
        assert report.skipped == 3
        assert ai_client.embedded_texts == []
        assert len(repository.stored) == 3

    async def test_duplicates_inside_one_batch_are_embedded_once(self) -> None:
        ai_client = FakeAIClient()
        duplicated = [*bullets(), bullets()[0]]

        report = await LoreIngestionService(InMemoryLoreChunkRepository(), ai_client).ingest(
            duplicated
        )

        assert report.stored == 3
        assert report.skipped == 1
        assert ai_client.embedded_texts == [
            "NO CURE",
            "MERCY HOSPITAL IS OVERRUN",
            "Tank has 4000 HP.",
        ]

    async def test_embeddings_travel_in_one_batch(self) -> None:
        """One request per record would make a corpus of hundreds needlessly slow."""
        ai_client = FakeAIClient()

        await LoreIngestionService(InMemoryLoreChunkRepository(), ai_client).ingest(bullets())

        assert ai_client.embedded_texts == [
            "NO CURE",
            "MERCY HOSPITAL IS OVERRUN",
            "Tank has 4000 HP.",
        ]

    async def test_empty_input_never_calls_the_provider(self) -> None:
        ai_client = FakeAIClient()
        repository = InMemoryLoreChunkRepository()

        report = await LoreIngestionService(repository, ai_client).ingest([])

        assert report.total == 0
        assert ai_client.embedded_texts == []
        assert repository.stored == []


class TestCorpusSnapshot:
    def test_round_trips_every_field(self, tmp_path: Path) -> None:
        snapshot = CorpusSnapshot(tmp_path / "corpus" / "lore.jsonl")

        snapshot.write(bullets())
        restored = snapshot.read()

        assert [record.model_dump() for record in restored] == [
            record.model_dump() for record in bullets()
        ]

    def test_summarises_records_per_tier(self, tmp_path: Path) -> None:
        snapshot = CorpusSnapshot(tmp_path / "lore.jsonl")
        snapshot.write(bullets())

        assert snapshot.summarise() == {"graffiti": 2, "official": 1}

    def test_missing_snapshot_points_at_the_command_that_builds_it(self, tmp_path: Path) -> None:
        snapshot = CorpusSnapshot(tmp_path / "absent.jsonl")

        with pytest.raises(ApplicationError, match="--refresh"):
            snapshot.read()

    def test_malformed_line_is_rejected_with_its_number(self, tmp_path: Path) -> None:
        path = tmp_path / "lore.jsonl"
        record = '{"content": "ok", "source_type": "official", "topic": "infected", "title": "t"}'
        path.write_text(f"{record}\n", encoding="utf-8")
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"content": "broken"\n')

        with pytest.raises(ApplicationError, match="line 2"):
            CorpusSnapshot(path).read()

    def test_written_file_is_one_json_object_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "lore.jsonl"
        CorpusSnapshot(path).write(bullets())

        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        assert len(lines) == 3
        assert all(isinstance(json.loads(line), dict) for line in lines)


class TestCorpusTranslator:
    async def test_rewrites_the_content_of_every_record(self) -> None:
        ai_client = FakeAIClient()

        outcome = await CorpusTranslator(ai_client, target_language="ru").translate(bullets())

        assert [document.content for document in outcome.documents] == [
            "[ru] NO CURE",
            "[ru] MERCY HOSPITAL IS OVERRUN",
            "[ru] Tank has 4000 HP.",
        ]
        assert outcome.translated == 3
        assert outcome.unchanged == 0

    async def test_keeps_metadata_of_the_translated_record(self) -> None:
        outcome = await CorpusTranslator(FakeAIClient(), target_language="ru").translate(bullets())

        assert outcome.documents[0].source_type is SourceType.GRAFFITI
        assert outcome.documents[0].topic is LoreTopic.GRAFFITI

    async def test_an_empty_translation_falls_back_to_the_source_text(self) -> None:
        class BlankTranslator(FakeAIClient):
            async def translate_texts(self, texts: Sequence[str]) -> list[str]:
                return ["" for _ in texts]

        translator = CorpusTranslator(BlankTranslator(), target_language="ru")

        outcome = await translator.translate(bullets())

        assert [document.content for document in outcome.documents] == [
            record.content for record in bullets()
        ]
        assert outcome.translated == 0
        assert outcome.unchanged == 3

    async def test_nothing_to_translate_returns_nothing(self) -> None:
        outcome = await CorpusTranslator(FakeAIClient(), target_language="ru").translate([])

        assert outcome.total == 0
        assert outcome.documents == []
