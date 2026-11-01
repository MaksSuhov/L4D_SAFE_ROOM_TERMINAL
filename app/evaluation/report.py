"""Сводка по прогону: средние значения и человекочитаемый отчёт.

Класс :class:`EvaluationReport` — только агрегатор: он не ходит ни в сеть, ни в
базу, ему на вход дают готовые :class:`~app.evaluation.domain.CaseResult`, а он
умеет посчитать средние, отдать словарь для JSON и напечатать текст.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.evaluation.domain import CaseResult

QUESTION_WIDTH: int = 30
"""Сколько символов вопроса показывать в таблице по кейсам."""


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Результаты прогона вместе со средними значениями."""

    results: tuple[CaseResult, ...]

    @property
    def count(self) -> int:
        """Сколько кейсов в отчёте."""
        return len(self.results)

    @property
    def mean_accuracy(self) -> float:
        """Средняя точность по судье (1–5)."""
        return _mean(result.judge.accuracy for result in self.results)

    @property
    def mean_completeness(self) -> float:
        """Средняя полнота по судье (1–5)."""
        return _mean(result.judge.completeness for result in self.results)

    @property
    def mean_relevance(self) -> float:
        """Средняя релевантность по судье (1–5)."""
        return _mean(result.judge.relevance for result in self.results)

    @property
    def mean_answer_precision(self) -> float:
        """Средняя precision ответа относительно эталона (0–1)."""
        return _mean(result.answer_overlap.precision for result in self.results)

    @property
    def mean_answer_recall(self) -> float:
        """Средняя recall ответа относительно эталона (0–1)."""
        return _mean(result.answer_overlap.recall for result in self.results)

    @property
    def mean_answer_f1(self) -> float:
        """Средний F1 ответа относительно эталона (0–1)."""
        return _mean(result.answer_overlap.f1 for result in self.results)

    @property
    def mean_source_precision(self) -> float:
        """Средняя precision поиска источников (0–1)."""
        return _mean(result.source_overlap.precision for result in self.results)

    @property
    def mean_source_recall(self) -> float:
        """Средняя recall поиска источников (0–1)."""
        return _mean(result.source_overlap.recall for result in self.results)

    @property
    def mean_source_f1(self) -> float:
        """Средний F1 поиска источников (0–1)."""
        return _mean(result.source_overlap.f1 for result in self.results)

    @property
    def mean_time_to_first_token(self) -> float:
        """Среднее время до первого токена в секундах."""
        return _mean(result.timing.time_to_first_token for result in self.results)

    @property
    def mean_total_time(self) -> float:
        """Среднее полное время ответа в секундах."""
        return _mean(result.timing.total for result in self.results)

    def to_dict(self) -> dict[str, object]:
        """Отчёт в виде словаря — для записи в JSON."""
        return {
            "summary": {
                "count": self.count,
                "judge": {
                    "accuracy": round(self.mean_accuracy, 4),
                    "completeness": round(self.mean_completeness, 4),
                    "relevance": round(self.mean_relevance, 4),
                },
                "answer_overlap": {
                    "precision": round(self.mean_answer_precision, 4),
                    "recall": round(self.mean_answer_recall, 4),
                    "f1": round(self.mean_answer_f1, 4),
                },
                "source_overlap": {
                    "precision": round(self.mean_source_precision, 4),
                    "recall": round(self.mean_source_recall, 4),
                    "f1": round(self.mean_source_f1, 4),
                },
                "timing": {
                    "time_to_first_token": round(self.mean_time_to_first_token, 4),
                    "total": round(self.mean_total_time, 4),
                },
            },
            "cases": [self._case_to_dict(result) for result in self.results],
        }

    def render_text(self) -> str:
        """Отчёт для чтения человеком в терминале."""
        lines = [
            "",
            f"Оценка терминала: {self.count} кейс(ов)",
            "",
            "Средние оценки судьи (1–5)",
            f"  Точность      : {self.mean_accuracy:.2f}",
            f"  Полнота       : {self.mean_completeness:.2f}",
            f"  Релевантность : {self.mean_relevance:.2f}",
            "",
            "Средние метрики пересечения ответа с эталоном (0–1)",
            f"  Precision : {self.mean_answer_precision:.2f}",
            f"  Recall    : {self.mean_answer_recall:.2f}",
            f"  F1        : {self.mean_answer_f1:.2f}",
            "",
            "Средняя точность поиска источников (0–1)",
            f"  Precision : {self.mean_source_precision:.2f}",
            f"  Recall    : {self.mean_source_recall:.2f}",
            f"  F1        : {self.mean_source_f1:.2f}",
            "",
            "Время ответа",
            f"  До первого токена : {self.mean_time_to_first_token:.3f} с",
            f"  Полное время      : {self.mean_total_time:.3f} с",
            "",
            "По кейсам",
            "  #    вопрос                          точн полн рел   P     R     F1    TTFT   время",
        ]
        for position, result in enumerate(self.results, start=1):
            lines.append(
                f"  {position:<4} {_shorten(result.case.question):<32} "
                f"{result.judge.accuracy:>4.0f} {result.judge.completeness:>4.0f} "
                f"{result.judge.relevance:>3.0f} "
                f"{result.answer_overlap.precision:>6.2f} {result.answer_overlap.recall:>5.2f} "
                f"{result.answer_overlap.f1:>5.2f} "
                f"{result.timing.time_to_first_token:>5.2f}с {result.timing.total:>5.2f}с"
            )
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _case_to_dict(result: CaseResult) -> dict[str, object]:
        """Один кейс в виде словаря — для записи в JSON."""
        return {
            "question": result.case.question,
            "character_id": result.case.character_id,
            "tags": list(result.case.tags),
            "answer": result.answer,
            "judge": {
                "accuracy": result.judge.accuracy,
                "completeness": result.judge.completeness,
                "relevance": result.judge.relevance,
                "comment": result.judge.comment,
            },
            "answer_overlap": {
                "precision": round(result.answer_overlap.precision, 4),
                "recall": round(result.answer_overlap.recall, 4),
                "f1": round(result.answer_overlap.f1, 4),
            },
            "source_overlap": {
                "precision": round(result.source_overlap.precision, 4),
                "recall": round(result.source_overlap.recall, 4),
                "f1": round(result.source_overlap.f1, 4),
            },
            "timing": {
                "time_to_first_token": round(result.timing.time_to_first_token, 4),
                "total": round(result.timing.total, 4),
            },
        }


def _mean(values: Iterable[float]) -> float:
    """Среднее по значениям; ноль, если значений нет."""
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _shorten(text: str) -> str:
    """Обрезать длинный вопрос, чтобы строки таблицы не разъезжались."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= QUESTION_WIDTH:
        return collapsed
    return collapsed[: QUESTION_WIDTH - 1] + "…"
