"""Проверки трейсинга: выключенный трейсер и разметка конвейера.

Сервисы размечают работу через порт, поэтому тесты проверяют разметку
двойником трейсера — без ключей, сети и SDK наблюдаемости.
"""

from __future__ import annotations

from app.interfaces import RetrievedChunk
from app.services import HideQueryExpander, NoOpTracer
from app.services.prompt_builder import ArchivePromptBuilder
from app.services.rag_service import RAGService
from tests.fakes import (
    FakeAIClient,
    InMemoryCharacterRepository,
    InMemoryLoreChunkRepository,
    RecordingTracer,
    build_character,
    build_chunk,
)

ANSWER = "Ищи в Атриуме."


def build_service(tracer: RecordingTracer) -> RAGService:
    """Собрать конвейер на двойниках, подключив записывающий трейсер."""
    chunk = build_chunk(chunk_id=1, content="Дробовик лежит на карте «Атриум».")
    repository = InMemoryLoreChunkRepository([RetrievedChunk(chunk=chunk, similarity=0.9)])
    return RAGService(
        InMemoryCharacterRepository([build_character()]),
        repository,
        ArchivePromptBuilder(),
        FakeAIClient(answer=ANSWER),
        tracer=tracer,
    )


async def test_a_chat_turn_opens_a_chat_span_that_carries_the_answer() -> None:
    tracer = RecordingTracer()
    service = build_service(tracer)

    await service.process_chat_message(1, "где дробовик?")

    assert tracer.names == ["chat", "retrieve"]
    assert tracer.named("chat").output == ANSWER


async def test_the_retrieval_span_carries_the_records_it_found() -> None:
    tracer = RecordingTracer()
    service = build_service(tracer)

    await service.process_chat_message(1, "где дробовик?")

    metadata = tracer.named("retrieve").metadata
    assert metadata["probes"] == ["где дробовик?"]
    assert metadata["official"] == [(1, 0.9)]
    assert metadata["graffiti"] == []


async def test_the_hide_expander_opens_its_own_span() -> None:
    tracer = RecordingTracer()
    expander = HideQueryExpander(
        FakeAIClient(completion='["Оружие на карте «Атриум»"]'),
        tracer=tracer,
    )

    expanded = await expander.expand("где дробовик в аттриуме?")

    assert tracer.names == ["hide-expand"]
    assert tracer.named("hide-expand").output == expanded.hypotheticals


async def test_the_null_tracer_swallows_annotation() -> None:
    tracer = NoOpTracer()

    with tracer.span("probe", as_type="retriever") as span:
        span.annotate(input="вопрос", output="ответ", metadata={"k": 1})

    tracer.flush()
    tracer.shutdown()
