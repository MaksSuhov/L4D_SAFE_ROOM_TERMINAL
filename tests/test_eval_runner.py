"""Tests for the evaluation runner, driven entirely by fakes.

The whole run is exercised without a database, a provider or a network: an
in-memory repository feeds the real ``RAGService``, a stubbed model streams the
answer, a stub judge returns fixed scores and a fake clock makes the timings
exact.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from app.evaluation import CaseEvaluator, EvalCase, EvaluationService
from app.evaluation.dataset import load_cases
from app.evaluation.domain import CaseResult, JudgeScore
from app.evaluation.latency import measure_stream
from app.interfaces import RetrievedChunk
from app.services import ArchivePromptBuilder, RAGService
from app.taxonomy import SourceType
from tests.fakes import (
    FakeAIClient,
    FakeClock,
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
    """Judge that returns the same verdict for every case and records the calls."""

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
    fragments: list[str],
    judge: StubJudge,
    clock: FakeClock | None = None,
) -> tuple[CaseEvaluator, FakeAIClient]:
    """Assemble the real RAG pipeline over fakes and wrap it in an evaluator."""
    model = FakeAIClient(stream_fragments=fragments)
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
        clock=clock or FakeClock(),
    )
    return evaluator, model


async def empty_stream() -> AsyncIterator[str]:
    """An async iterator that produces nothing at all."""
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

        assert result.answer_overlap.f1 == 1.0

    async def test_scores_the_retrieved_sources(self) -> None:
        evaluator, _ = build_evaluator(fragments=["ответ"], judge=StubJudge())

        result = await evaluator.evaluate(CASE)

        assert result.source_overlap.precision == 1.0
        assert result.source_overlap.recall == 1.0

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
        assert summary["timing"] == {"time_to_first_token": 1.0, "total": 2.0}
        assert len(cases) == 1

    async def test_renders_a_readable_text_report(self) -> None:
        evaluator, _ = build_evaluator(fragments=["Танк — мутант."], judge=StubJudge())
        report = await EvaluationService(evaluator).run([CASE])

        text = report.render_text()

        assert "Точность" in text
        assert "До первого токена" in text
        assert "Кто такой Танк?" in text


class TestMeasureStream:
    async def test_an_empty_stream_times_the_wait_for_nothing(self) -> None:
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
