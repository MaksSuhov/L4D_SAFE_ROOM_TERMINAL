from __future__ import annotations

from dataclasses import dataclass, field

SCORE_MIN: float = 1.0
SCORE_MAX: float = 5.0


@dataclass(frozen=True, slots=True)
class EvalCase:

    question: str
    character_id: int
    reference_answer: str = ""
    expected_sources: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class JudgeScore:

    accuracy: float
    completeness: float
    relevance: float
    comment: str = ""


@dataclass(frozen=True, slots=True)
class Overlap:

    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class SourceOverlap:
    """Насколько выборка нашла ожидаемые темы и на каких местах.

    ``precision`` и ``recall`` отвечают на вопрос «нашлось ли», но о порядке не
    говорят ничего, а для поиска с фиксированным бюджетом это разные вещи.
    ``full_hit`` — все ожидаемые темы нашлись (1.0) или нет (0.0); ``mrr`` —
    обратное место первой найденной темы: единица, если она стоит первой в выдаче.
    """

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    full_hit: float = 0.0
    mrr: float = 0.0


@dataclass(frozen=True, slots=True)
class Timing:

    time_to_first_token: float
    total: float


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Всё, что удалось измерить по одному вопросу."""

    case: EvalCase
    answer: str
    timing: Timing
    judge: JudgeScore
    lexical_overlap: Overlap
    semantic_similarity: float
    source_overlap: SourceOverlap
    sources: tuple[str, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class CaseFailure:
    """Кейс, который так и не дал ответа.

    Сбой — часть результата прогона, а не пропуск в нём: вопрос, оставшийся без
    ответа, обязан быть виден в отчёте, иначе средние считались бы только по
    выжившим кейсам и были бы завышены ровно на сбойных.
    """

    case: EvalCase
    reason: str
