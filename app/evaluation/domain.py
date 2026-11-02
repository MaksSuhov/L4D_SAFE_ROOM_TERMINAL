from __future__ import annotations

from dataclasses import dataclass

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
    answer_overlap: Overlap
    source_overlap: Overlap
