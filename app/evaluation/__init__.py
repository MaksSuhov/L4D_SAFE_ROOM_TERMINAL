"""Оценка качества ответов терминала.

Пакет собран слоями, у каждого — одна ответственность:

* :mod:`~app.evaluation.domain` — значения (кейс, баллы, замеры, результат);
* :mod:`~app.evaluation.interfaces` — порты, от которых зависят шаги;
* :mod:`~app.evaluation.metrics` — чистые формулы precision/recall/F1;
* :mod:`~app.evaluation.judge` — LLM-судья: точность, полнота, релевантность;
* :mod:`~app.evaluation.latency` — часы и замер времени до первого токена;
* :mod:`~app.evaluation.dataset` — чтение кейсов из JSONL;
* :mod:`~app.evaluation.runner` — прогон одного кейса и всего датасета;
* :mod:`~app.evaluation.report` — средние значения и человекочитаемый отчёт.

Наружу пакет отдаёт готовый набор имён, поэтому вызывающему коду достаточно
одного импорта::

    from app.evaluation import CaseEvaluator, EvaluationService, LLMJudge, load_cases
"""

from app.evaluation.dataset import build_case, load_cases
from app.evaluation.domain import CaseResult, EvalCase, JudgeScore, Overlap, Timing
from app.evaluation.judge import LLMJudge
from app.evaluation.latency import MonotonicClock, measure_stream
from app.evaluation.metrics import set_overlap, text_overlap, tokenize
from app.evaluation.report import EvaluationReport
from app.evaluation.runner import CaseEvaluator, EvaluationService

__all__ = [
    "CaseEvaluator",
    "CaseResult",
    "EvalCase",
    "EvaluationReport",
    "EvaluationService",
    "JudgeScore",
    "LLMJudge",
    "MonotonicClock",
    "Overlap",
    "Timing",
    "build_case",
    "load_cases",
    "measure_stream",
    "set_overlap",
    "text_overlap",
    "tokenize",
]
