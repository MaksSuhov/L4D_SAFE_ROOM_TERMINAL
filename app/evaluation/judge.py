"""LLM-судья: модель оценивает ответ и возвращает разобранные баллы.

Класс зависит только от порта :class:`~app.evaluation.interfaces.CompletionProtocol`,
то есть от «умения получить текстовый ответ». Ему не нужны ни эмбеддинги, ни
поиск, ни персонажи, — поэтому в тестах он проверяется обычной заглушкой.
"""

from __future__ import annotations

import json
import re

from app.evaluation.domain import SCORE_MAX, SCORE_MIN, JudgeScore
from app.evaluation.interfaces import CompletionProtocol
from app.exceptions import EvaluationError

JUDGE_SYSTEM_PROMPT: str = (
    "Ты — строгий, но справедливый проверяющий ответов игрового терминала про "
    "Left 4 Dead. Тебе дают вопрос игрока, ответ терминала, эталонный ответ "
    "(может отсутствовать) и выдержки из архива, которыми разрешалось "
    "пользоваться.\n\n"
    "Оцени ответ целым числом от 1 до 5 по трём критериям:\n"
    "- accuracy — точность: нет ли выдуманных фактов, не противоречит ли ответ "
    "эталону и архиву;\n"
    "- completeness — полнота: раскрыты ли все ключевые моменты эталона;\n"
    "- relevance — релевантность: отвечает ли ответ именно на заданный вопрос.\n\n"
    "Если эталонного ответа нет, считай эталоном выдержки из архива. "
    "Отвечай ТОЛЬКО одним JSON-объектом без markdown и без пояснений вокруг:\n"
    '{"accuracy": 1-5, "completeness": 1-5, "relevance": 1-5, '
    '"comment": "короткое объяснение по-русски"}'
)

JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class LLMJudge:
    """Оценивает один ответ через обычный текстовый запрос к модели."""

    def __init__(self, model: CompletionProtocol) -> None:
        self._model = model

    async def evaluate(
        self,
        *,
        question: str,
        answer: str,
        reference: str = "",
        context: str = "",
    ) -> JudgeScore:
        """Спросить у модели баллы и разобрать её ответ."""
        reply = await self._model.complete(
            JUDGE_SYSTEM_PROMPT,
            self._build_request(
                question=question,
                answer=answer,
                reference=reference,
                context=context,
            ),
            temperature=0.0,
        )
        return self._parse(reply)

    @staticmethod
    def _build_request(*, question: str, answer: str, reference: str, context: str) -> str:
        """Собрать текст задания для судьи."""
        return "\n\n".join(
            (
                f"Вопрос игрока:\n{question}",
                f"Ответ терминала:\n{answer}",
                f"Эталонный ответ:\n{reference or '(не задан)'}",
                f"Выдержки из архива:\n{context or '(пусто)'}",
            )
        )

    @staticmethod
    def _parse(reply: str) -> JudgeScore:
        """Достать баллы из ответа модели, выдержав мусор вокруг JSON."""
        payload = LLMJudge._extract_object(reply)
        return JudgeScore(
            accuracy=_clamp(payload.get("accuracy"), "accuracy"),
            completeness=_clamp(payload.get("completeness"), "completeness"),
            relevance=_clamp(payload.get("relevance"), "relevance"),
            comment=str(payload.get("comment", "")).strip(),
        )

    @staticmethod
    def _extract_object(reply: str) -> dict[str, object]:
        """Найти в ответе первый JSON-объект."""
        match = JSON_OBJECT_PATTERN.search(reply)
        if match is None:
            raise EvaluationError(f"Judge reply did not contain a JSON object: {reply!r}")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise EvaluationError(f"Judge reply was not valid JSON: {error}") from error
        if not isinstance(parsed, dict):
            raise EvaluationError("Judge reply was not a JSON object")
        return parsed


def _clamp(value: object, field: str) -> float:
    """Превратить значение в число и вписать его в границы шкалы."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise EvaluationError(f"Judge returned a non-numeric {field}: {value!r}") from error
    return min(SCORE_MAX, max(SCORE_MIN, number))
