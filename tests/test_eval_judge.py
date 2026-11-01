"""Tests for the LLM judge, running against a stub completion model."""

from __future__ import annotations

import pytest

from app.evaluation.judge import JUDGE_SYSTEM_PROMPT, LLMJudge
from app.exceptions import EvaluationError
from tests.fakes import FakeAIClient


def judge_with(reply: str) -> tuple[LLMJudge, FakeAIClient]:
    """Build a judge whose model always answers with ``reply``."""
    model = FakeAIClient(completion=reply)
    return LLMJudge(model), model


class TestJudge:
    async def test_parses_the_three_criteria(self) -> None:
        judge, _ = judge_with(
            '{"accuracy": 5, "completeness": 4, "relevance": 3, "comment": "хорошо"}'
        )

        score = await judge.evaluate(question="Кто такой Танк?", answer="Танк — мутант.")

        assert (score.accuracy, score.completeness, score.relevance) == (5.0, 4.0, 3.0)
        assert score.comment == "хорошо"

    async def test_tolerates_prose_around_the_json(self) -> None:
        judge, _ = judge_with(
            'Разбор:\n{"accuracy": 4, "completeness": 4, "relevance": 4}\nГотово.'
        )

        score = await judge.evaluate(question="q", answer="a")

        assert score.relevance == 4.0

    async def test_clamps_scores_into_the_scale(self) -> None:
        judge, _ = judge_with('{"accuracy": 9, "completeness": 0, "relevance": 3}')

        score = await judge.evaluate(question="q", answer="a")

        assert score.accuracy == 5.0
        assert score.completeness == 1.0

    async def test_asks_for_determinism(self) -> None:
        judge, model = judge_with('{"accuracy": 1, "completeness": 1, "relevance": 1}')

        await judge.evaluate(question="q", answer="a")

        assert model.completion_calls[0]["temperature"] == 0.0
        assert model.completion_calls[0]["system_prompt"] == JUDGE_SYSTEM_PROMPT

    async def test_the_request_carries_the_question_and_the_answer(self) -> None:
        judge, model = judge_with('{"accuracy": 1, "completeness": 1, "relevance": 1}')

        await judge.evaluate(
            question="Кто такой Танк?",
            answer="Мутант.",
            reference="Эталон.",
            context="Выдержка.",
        )

        request = str(model.completion_calls[0]["user_message"])
        assert "Кто такой Танк?" in request
        assert "Мутант." in request
        assert "Эталон." in request

    async def test_a_reply_without_json_is_rejected(self) -> None:
        judge, _ = judge_with("Не могу оценить.")

        with pytest.raises(EvaluationError, match="JSON"):
            await judge.evaluate(question="q", answer="a")

    async def test_a_non_numeric_score_is_rejected(self) -> None:
        judge, _ = judge_with('{"accuracy": "хорошо", "completeness": 3, "relevance": 3}')

        with pytest.raises(EvaluationError, match="non-numeric accuracy"):
            await judge.evaluate(question="q", answer="a")
