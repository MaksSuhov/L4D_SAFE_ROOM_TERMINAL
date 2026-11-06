"""Тесты прогона оценки, целиком работающие на заглушках.

Весь прогон проверяется без базы, провайдера и сети: репозиторий в памяти
питает настоящий ``RAGService``, заглушка модели стримит ответ, заглушка судьи
возвращает фиксированные баллы, а поддельные часы делают замеры точными.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.evaluation import CaseEvaluator, EvalCase, EvaluationService
from app.evaluation.dataset import load_cases
from app.evaluation.domain import CaseResult, JudgeScore
from app.evaluation.latency import measure_stream
from app.exceptions import AIServiceError, EvaluationError
from app.interfaces import RetrievedChunk
from app.services import ArchivePromptBuilder, RAGService
from app.taxonomy import SourceType
from tests.fakes import (
    FakeAIClient,
    FakeClock,
    FakeSimilarity,
    InMemoryCharacterRepository,
    InMemoryLoreChunkRepository,
    build_character,
    build_chunk,
)

VERDICT = JudgeScore(accuracy=5.0, completeness=4.0, relevance=5.0, comment="по делу")

CASE = EvalCase(
    question="Кто такой Танк?",
    character_id=5,
    reference_answer="Танк мутант",
    expected_sources=("Танк",),
    tags=("infected",),
)


class StubJudge:
    """Судья, возвращающий один и тот же вердикт для каждого кейса и записывающий вызовы."""

    def __init__(self, verdict: JudgeScore = VERDICT) -> None:
        self._verdict = verdict
        self.calls: list[dict[str, str]] = []

    async def evaluate(
        self,
        *,
        question: str,
        answer: str,
        reference: str = "",
        context: str = "",
    ) -> JudgeScore:
        self.calls.append(
            {"question": question, "answer": answer, "reference": reference, "context": context}
        )
        return self._verdict


def build_evaluator(
    *,
    judge: StubJudge,
    fragments: list[str] | None = None,
    sequences: list[list[str]] | None = None,
    errors: list[Exception] | None = None,
    similarity: FakeSimilarity | None = None,
    clock: FakeClock | None = None,
    attempts: int = 1,
    temperature: float | None = None,
) -> tuple[CaseEvaluator, FakeAIClient]:
    """Собрать настоящий конвейер RAG на заглушках и обернуть его в оценщик."""
    model = FakeAIClient(
        stream_fragments=fragments if fragments is not None else ["ответ"],
        stream_sequences=sequences,
        stream_errors=errors,
    )
    rag_service = RAGService(
        InMemoryCharacterRepository(
            [build_character(character_id=5, name="Билл", system_prompt="Ты — Билл.")]
        ),
        InMemoryLoreChunkRepository(
            [
                RetrievedChunk(
                    chunk=build_chunk(
                        chunk_id=10,
                        content="Танк швыряет камни.",
                        source_type=SourceType.OFFICIAL,
                        title="Танк › Базовые стратегии",
                    ),
                    similarity=0.9,
                )
            ]
        ),
        ArchivePromptBuilder(),
        model,
    )
    evaluator = CaseEvaluator(
        context_builder=rag_service,
        model=model,
        judge=judge,
        similarity=similarity,
        clock=clock or FakeClock(),
        attempts=attempts,
        temperature=temperature,
    )
    return evaluator, model


async def empty_stream() -> AsyncIterator[str]:
    """Асинхронный итератор, не порождающий ничего."""
    fragments: list[str] = []
    for fragment in fragments:
        yield fragment


class TestCaseEvaluator:
    async def test_measures_the_time_to_the_first_token(self) -> None:
        evaluator, _ = build_evaluator(
            fragments=["Танк ", "— ", "мутант."],
            judge=StubJudge(),
            clock=FakeClock(start=0.0, step=1.0),
        )

        result = await evaluator.evaluate(CASE)

        assert result.answer == "Танк — мутант."
        assert result.timing.time_to_first_token == 1.0
        assert result.timing.total == 2.0

    async def test_scores_the_answer_against_the_reference(self) -> None:
        evaluator, _ = build_evaluator(
            fragments=["Танк — мутант."],
            judge=StubJudge(),
            clock=FakeClock(start=0.0, step=0.5),
        )

        result = await evaluator.evaluate(CASE)

        assert result.lexical_overlap.f1 == 1.0

    async def test_scores_the_retrieved_sources(self) -> None:
        evaluator, _ = build_evaluator(fragments=["ответ"], judge=StubJudge())

        result = await evaluator.evaluate(CASE)

        assert result.source_overlap.precision == 1.0
        assert result.source_overlap.recall == 1.0
        assert result.source_overlap.full_hit == 1.0

    async def test_records_the_semantic_similarity(self) -> None:
        similarity = FakeSimilarity(value=0.8)
        evaluator, _ = build_evaluator(fragments=["Танк — мутант."], judge=StubJudge(),
                                       similarity=similarity)

        result = await evaluator.evaluate(CASE)

        assert result.semantic_similarity == 0.8
        assert similarity.calls == [("Танк мутант", "Танк — мутант.")]

    async def test_passes_the_pinned_temperature_to_generation(self) -> None:
        evaluator, model = build_evaluator(fragments=["ответ"], judge=StubJudge(),
                                           temperature=0.0)

        await evaluator.evaluate(CASE)

        assert model.stream_temperatures == [0.0]

    async def test_the_pipeline_really_ran_before_generation(self) -> None:
        evaluator, model = build_evaluator(fragments=["ответ"], judge=StubJudge())

        await evaluator.evaluate(CASE)

        context = str(model.generation_calls[0]["context"])
        assert "official_lore" in context
        assert "Танк швыряет камни." in context

    async def test_the_judge_receives_the_generated_answer(self) -> None:
        judge = StubJudge()
        evaluator, _ = build_evaluator(fragments=["Танк — мутант."], judge=judge)

        await evaluator.evaluate(CASE)

        assert judge.calls[0]["question"] == "Кто такой Танк?"
        assert judge.calls[0]["answer"] == "Танк — мутант."
        assert judge.calls[0]["reference"] == "Танк мутант"

    async def test_retries_an_empty_answer(self) -> None:
        """Проверить, что пустая выдача провайдера не принимается за ответ."""
        evaluator, _ = build_evaluator(
            sequences=[[], ["Танк — мутант."]],
            judge=StubJudge(),
            attempts=2,
        )

        result = await evaluator.evaluate(CASE)

        assert result.answer == "Танк — мутант."

    async def test_retries_an_attempt_that_failed(self) -> None:
        evaluator, _ = build_evaluator(
            fragments=["Танк — мутант."],
            errors=[AIServiceError("stream returned no content")],
            judge=StubJudge(),
            attempts=2,
        )

        result = await evaluator.evaluate(CASE)

        assert result.answer == "Танк — мутант."

    async def test_gives_up_after_the_attempts_run_out(self) -> None:
        evaluator, _ = build_evaluator(sequences=[[], []], judge=StubJudge(), attempts=2)

        with pytest.raises(EvaluationError, match="no answer"):
            await evaluator.evaluate(CASE)


class TestEvaluationService:
    async def test_reports_progress_for_every_case(self) -> None:
        evaluator, _ = build_evaluator(fragments=["ответ"], judge=StubJudge())
        seen: list[str] = []

        def remember(result: CaseResult) -> None:
            seen.append(result.answer)

        report = await EvaluationService(evaluator).run([CASE, CASE], on_result=remember)

        assert seen == ["ответ", "ответ"]
        assert report.count == 2

    async def test_average_of_the_judge_scores(self) -> None:
        evaluator, _ = build_evaluator(fragments=["Танк — мутант."], judge=StubJudge())

        report = await EvaluationService(evaluator).run([CASE])

        assert report.mean_accuracy == 5.0
        assert report.mean_completeness == 4.0
        assert report.mean_relevance == 5.0

    async def test_an_empty_run_reports_zeroes(self) -> None:
        evaluator, _ = build_evaluator(fragments=["ответ"], judge=StubJudge())

        report = await EvaluationService(evaluator).run([])

        assert report.count == 0
        assert report.mean_accuracy == 0.0
        assert report.mean_total_time == 0.0

    async def test_a_case_without_an_answer_becomes_a_failure(self) -> None:
        """Проверить, что сбойный кейс попадает в отчёт, а не выпадает из него."""
        evaluator, _ = build_evaluator(sequences=[[], []], judge=StubJudge(), attempts=2)

        report = await EvaluationService(evaluator).run([CASE])

        assert report.count == 0
        assert report.failed_count == 1
        assert report.failures[0].case.question == "Кто такой Танк?"

    async def test_the_judge_is_never_asked_about_an_empty_answer(self) -> None:
        judge = StubJudge()
        evaluator, _ = build_evaluator(sequences=[[]], judge=judge, attempts=1)

        report = await EvaluationService(evaluator).run([CASE])

        assert report.failed_count == 1
        assert judge.calls == []


class TestEvaluationReport:
    async def test_serialises_to_a_summary_and_per_case_rows(self) -> None:
        evaluator, _ = build_evaluator(
            fragments=["Танк — мутант."],
            judge=StubJudge(),
            clock=FakeClock(start=0.0, step=1.0),
        )
        report = await EvaluationService(evaluator).run([CASE])

        payload = report.to_dict()

        summary = payload["summary"]
        cases = payload["cases"]
        assert isinstance(summary, dict)
        assert isinstance(cases, list)
        assert summary["count"] == 1
        assert summary["failed"] == 0
        assert summary["timing"] == {"time_to_first_token": 1.0, "total": 2.0}
        assert len(cases) == 1
        assert payload["failures"] == []

    async def test_renders_a_readable_text_report(self) -> None:
        evaluator, _ = build_evaluator(fragments=["Танк — мутант."], judge=StubJudge())
        report = await EvaluationService(evaluator).run([CASE])

        text = report.render_text()

        assert "Точность" in text
        assert "До первого токена" in text
        assert "Кто такой Танк?" in text

    async def test_lists_failures_apart_from_the_means(self) -> None:
        evaluator, _ = build_evaluator(sequences=[[]], judge=StubJudge(), attempts=1)
        report = await EvaluationService(evaluator).run([CASE])

        text = report.render_text()

        assert "Сбои" in text
        assert "Кто такой Танк?" in text


class TestMeasureStream:
    async def test_an_empty_stream_times_the_wait_for_nothing(self) -> None:
        """Проверить, что замер честно меряет пустоту; ошибкой её делает транспорт."""
        answer, timing = await measure_stream(empty_stream(), FakeClock(start=0.0, step=1.0))

        assert answer == ""
        assert timing.time_to_first_token == timing.total


class TestLoadCases:
    def test_reads_and_skips_comments(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.jsonl"
        path.write_text(
            "# a note\n"
            '{"question": "Танк?", "character_id": 5, "expected_sources": ["Танк"]}\n'
            "\n",
            encoding="utf-8",
        )

        cases = load_cases(path)

        assert len(cases) == 1
        assert cases[0].expected_sources == ("Танк",)
