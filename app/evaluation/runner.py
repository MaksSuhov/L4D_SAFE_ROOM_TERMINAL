"""Оркестрация оценки: прогнать кейсы и собрать отчёт.

Класс :class:`CaseEvaluator` отвечает за один вопрос, класс
:class:`EvaluationService` — за список. Оба зависят только от портов, поэтому
весь прогон можно выполнить на заглушках, без базы, сети и ключа.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from app.evaluation.domain import CaseResult, EvalCase
from app.evaluation.interfaces import (
    ClockProtocol,
    ContextBuilderProtocol,
    JudgeProtocol,
    StreamingChatProtocol,
)
from app.evaluation.latency import MonotonicClock, measure_stream
from app.evaluation.metrics import set_overlap, text_overlap
from app.evaluation.report import EvaluationReport

logger = logging.getLogger("l4d.eval")


class CaseEvaluator:
    """Оценивает один кейс: собирает контекст, стримит ответ, считает метрики."""

    def __init__(
        self,
        *,
        context_builder: ContextBuilderProtocol,
        model: StreamingChatProtocol,
        judge: JudgeProtocol,
        clock: ClockProtocol | None = None,
    ) -> None:
        self._context_builder = context_builder
        self._model = model
        self._judge = judge
        self._clock = clock or MonotonicClock()

    async def evaluate(self, case: EvalCase) -> CaseResult:
        """Прогнать один вопрос через терминал и оценить получившийся ответ."""
        prepared = await self._context_builder.build_chat_context(
            case.character_id, case.question
        )

        tokens = self._model.stream_response(
            system_prompt=prepared.system_prompt,
            context=prepared.context,
            user_message=case.question,
            few_shot_examples=prepared.few_shot_examples,
        )
        answer, timing = await measure_stream(tokens, self._clock)

        score = await self._judge.evaluate(
            question=case.question,
            answer=answer,
            reference=case.reference_answer,
            context=prepared.context,
        )

        return CaseResult(
            case=case,
            answer=answer,
            timing=timing,
            judge=score,
            answer_overlap=text_overlap(case.reference_answer, answer),
            source_overlap=set_overlap(
                case.expected_sources,
                [source.title for source in prepared.sources],
            ),
        )


class EvaluationService:
    """Прогоняет кейсы по очереди и собирает из результатов отчёт."""

    def __init__(self, evaluator: CaseEvaluator) -> None:
        self._evaluator = evaluator

    async def run(
        self,
        cases: Sequence[EvalCase],
        *,
        on_result: Callable[[CaseResult], None] | None = None,
    ) -> EvaluationReport:
        """Оценить все ``cases`` и вернуть сводный отчёт.

        Кейсы идут по очереди, а не параллельно: так прогон не упирается в
        лимиты провайдера, а замеры времени остаются честными. ``on_result``
        вызывается после каждого кейса — по нему удобно печатать прогресс.
        """
        results: list[CaseResult] = []
        for position, case in enumerate(cases, start=1):
            logger.info("Evaluating case %d/%d: %s", position, len(cases), case.question)
            result = await self._evaluator.evaluate(case)
            results.append(result)
            if on_result is not None:
                on_result(result)
        return EvaluationReport(tuple(results))
