from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from app.evaluation.domain import CaseFailure, CaseResult, EvalCase, Timing
from app.evaluation.interfaces import (
    ClockProtocol,
    ContextBuilderProtocol,
    JudgeProtocol,
    SimilarityProtocol,
    StreamingChatProtocol,
)
from app.evaluation.latency import MonotonicClock, measure_stream
from app.evaluation.metrics import source_overlap, text_overlap
from app.evaluation.report import EvaluationReport
from app.evaluation.similarity import NullSimilarity
from app.exceptions import AIServiceError, EvaluationError
from app.interfaces import PreparedChat

logger = logging.getLogger("l4d.eval")


class CaseEvaluator:
    """Оценивает один кейс: собирает контекст, стримит ответ, считает метрики."""

    def __init__(
        self,
        *,
        context_builder: ContextBuilderProtocol,
        model: StreamingChatProtocol,
        judge: JudgeProtocol,
        similarity: SimilarityProtocol | None = None,
        clock: ClockProtocol | None = None,
        attempts: int = 1,
        temperature: float | None = None,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        self._context_builder = context_builder
        self._model = model
        self._judge = judge
        self._similarity = similarity or NullSimilarity()
        self._clock = clock or MonotonicClock()
        self._attempts = attempts
        self._temperature = temperature

    async def evaluate(self, case: EvalCase) -> CaseResult:
        """Прогнать один вопрос через терминал и оценить получившийся ответ."""
        prepared = await self._context_builder.build_chat_context(
            case.character_id, case.question
        )

        answer, timing = await self._generate(case, prepared)

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
            lexical_overlap=text_overlap(case.reference_answer, answer),
            semantic_similarity=await self._similarity.similarity(
                case.reference_answer, answer
            ),
            source_overlap=source_overlap(
                case.expected_sources,
                [source.title for source in prepared.sources],
            ),
            sources=tuple(source.title for source in prepared.sources),
        )

    async def _generate(self, case: EvalCase, prepared: PreparedChat) -> tuple[str, Timing]:
        """Получить ответ, повторяя попытку после пустой выдачи или сбоя.

        Провайдер изредка отвечает потоком без единого фрагмента; принять это за
        ответ значило бы оценить пустоту. Кейс повторяется, а если ответа нет и
        после всех попыток — падает: сбой обязан попасть в отчёт отдельной строкой,
        а не раствориться в средних.
        """
        last_error: Exception | None = None
        for attempt in range(1, self._attempts + 1):
            try:
                answer, timing = await measure_stream(
                    self._model.stream_response(
                        system_prompt=prepared.system_prompt,
                        context=prepared.context,
                        user_message=case.question,
                        few_shot_examples=prepared.few_shot_examples,
                        temperature=self._temperature,
                    ),
                    self._clock,
                )
            except AIServiceError as error:
                last_error = error
                logger.warning(
                    "Case %r failed on attempt %d/%d: %s",
                    case.question,
                    attempt,
                    self._attempts,
                    error,
                )
                continue

            if answer.strip():
                return answer, timing

            last_error = EvaluationError("the model returned an empty answer")
            logger.warning(
                "Case %r came back empty on attempt %d/%d",
                case.question,
                attempt,
                self._attempts,
            )

        raise EvaluationError(
            f"Case {case.question!r} produced no answer after {self._attempts} attempt(s)"
        ) from last_error


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

        Кейсы идут по очереди, а не параллельно: так прогон не упирается в лимиты
        провайдера, а замеры времени остаются честными. ``on_result`` вызывается
        после каждого удачного кейса — по нему удобно печатать прогресс.

        Кейс, не давший ответа, не роняет прогон и не исчезает: он попадает в
        отчёт как сбой. Иначе один сбойный вопрос молча выпал бы из средних, и
        они стали бы тем оптимистичнее, чем хуже шёл прогон.
        """
        results: list[CaseResult] = []
        failures: list[CaseFailure] = []
        for position, case in enumerate(cases, start=1):
            logger.info("Evaluating case %d/%d: %s", position, len(cases), case.question)
            try:
                result = await self._evaluator.evaluate(case)
            except (AIServiceError, EvaluationError) as error:
                logger.warning("Case %d/%d failed: %s", position, len(cases), error)
                failures.append(CaseFailure(case=case, reason=str(error)))
                continue
            results.append(result)
            if on_result is not None:
                on_result(result)
        return EvaluationReport(tuple(results), tuple(failures))
