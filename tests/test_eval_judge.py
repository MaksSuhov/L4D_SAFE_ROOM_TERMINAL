"""Тесты LLM-судьи, работающие против заглушки модели дополнений."""

from __future__ import annotations

import pytest

from app.evaluation.judge import JUDGE_SYSTEM_PROMPT, LLMJudge
from app.exceptions import EvaluationError
from tests.fakes import FakeAIClient


def judge_with(reply: str) -> tuple[LLMJudge, FakeAIClient]:
    """Собрать судью, модель которого всегда отвечает ``reply``."""
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

    async def test_the_answer_is_placed_after_the_reference(self) -> None:
        """Проверить, что оцениваемый ответ идёт последним, а не первым.

        Слабая модель оценивает то, что лежит рядом с вопросом; когда ответ шёл
        первым, она нередко оценивала эталон вместо него.
        """
        judge, model = judge_with('{"accuracy": 1, "completeness": 1, "relevance": 1}')

        await judge.evaluate(question="q", answer="ответ терминала", reference="эталон")

        request = str(model.completion_calls[0]["user_message"])
        assert request.index("эталон") < request.index("ответ терминала")

    async def test_an_empty_answer_is_not_sent_to_the_model(self) -> None:
        """Проверить, что судья не спрашивает модель про пустой ответ.

        Спрошенная, она оценила бы справку рядом и выдала высокие баллы ни за что.
        """
        judge, model = judge_with('{"accuracy": 5, "completeness": 5, "relevance": 5}')

        score = await judge.evaluate(question="q", answer="", reference="эталон")

        assert (score.accuracy, score.completeness, score.relevance) == (1.0, 1.0, 1.0)
        assert model.completion_calls == []

    async def test_an_answer_without_any_words_is_not_sent_to_the_model(self) -> None:
        """Проверить, что ответ из одних знаков препинания — тоже пустой."""
        judge, model = judge_with('{"accuracy": 5, "completeness": 5, "relevance": 5}')

        score = await judge.evaluate(question="q", answer="  ...  ", reference="эталон")

        assert score.accuracy == 1.0
        assert model.completion_calls == []

    async def test_a_reply_without_json_is_rejected(self) -> None:
        judge, _ = judge_with("Не могу оценить.")

        with pytest.raises(EvaluationError, match="JSON"):
            await judge.evaluate(question="q", answer="a")

    async def test_a_non_numeric_score_is_rejected(self) -> None:
        judge, _ = judge_with('{"accuracy": "хорошо", "completeness": 3, "relevance": 3}')

        with pytest.raises(EvaluationError, match="non-numeric accuracy"):
            await judge.evaluate(question="q", answer="a")
