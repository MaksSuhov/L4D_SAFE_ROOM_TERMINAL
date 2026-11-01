"""Значения, которыми обмениваются шаги оценки.

Модуль ничего не знает ни про модель, ни про сеть: это просто контейнеры
данных. Из-за этого их легко печатать, сравнивать в тестах и складывать в
отчёт, не поднимая ни базы, ни провайдера.
"""

from __future__ import annotations

from dataclasses import dataclass

SCORE_MIN: float = 1.0
"""Нижняя граница оценки судьи: 1 — совсем плохо."""

SCORE_MAX: float = 5.0
"""Верхняя граница оценки судьи: 5 — отлично."""


@dataclass(frozen=True, slots=True)
class EvalCase:
    """Один вопрос из датасета и то, каким должен быть хороший ответ.

    ``reference_answer`` и ``expected_sources`` необязательны. Если эталона нет,
    судья опирается только на выдержки из архива, а метрика пересечения с
    эталоном не считается.
    """

    question: str
    character_id: int
    reference_answer: str = ""
    expected_sources: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class JudgeScore:
    """Баллы судьи по трём критериям и его короткое объяснение."""

    accuracy: float
    completeness: float
    relevance: float
    comment: str = ""


@dataclass(frozen=True, slots=True)
class Overlap:
    """Три метрики пересечения: точность, полнота и их среднее (F1)."""

    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class Timing:
    """Время ответа в секундах: до первого токена и до конца генерации."""

    time_to_first_token: float
    total: float


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Всё, что удалось измерить по одному вопросу."""

    case: EvalCase
    answer: str
    timing: Timing
    judge: JudgeScore
    answer_overlap: Overlap
    source_overlap: Overlap
