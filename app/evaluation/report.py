from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.evaluation.domain import CaseFailure, CaseResult

QUESTION_WIDTH: int = 30
"""Сколько символов вопроса показывать в таблице по кейсам."""


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Результаты прогона вместе со средними значениями.

    Средние считаются только по кейсам, давшим ответ; сбойные перечисляются
    отдельно и в арифметику не входят. Иначе пустой ответ тянул бы средние вниз
    сразу по двум причинам — как ноль по метрикам и как выпавшая попытка, — а
    причина сбоя оставалась бы невидимой.
    """

    results: tuple[CaseResult, ...]
    failures: tuple[CaseFailure, ...] = ()

    @property
    def count(self) -> int:
        """Сколько кейсов дали ответ и участвуют в средних."""
        return len(self.results)

    @property
    def failed_count(self) -> int:
        """Сколько кейсов остались без ответа."""
        return len(self.failures)

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
    def mean_lexical_precision(self) -> float:
        """Средняя precision ответа относительно эталона (0–1)."""
        return _mean(result.lexical_overlap.precision for result in self.results)

    @property
    def mean_lexical_recall(self) -> float:
        """Средняя recall ответа относительно эталона (0–1)."""
        return _mean(result.lexical_overlap.recall for result in self.results)

    @property
    def mean_lexical_f1(self) -> float:
        """Средний F1 ответа относительно эталона (0–1)."""
        return _mean(result.lexical_overlap.f1 for result in self.results)

    @property
    def mean_semantic_similarity(self) -> float:
        """Средняя семантическая близость ответа к эталону (0–1)."""
        return _mean(result.semantic_similarity for result in self.results)

    @property
    def mean_source_precision(self) -> float:
        """Средняя precision поиска источников (0–1)."""
        return _mean(result.source_overlap.precision for result in self.results)

    @property
    def mean_source_recall(self) -> float:
        """Средняя полнота поиска источников (0–1)."""
        return _mean(result.source_overlap.recall for result in self.results)

    @property
    def mean_source_f1(self) -> float:
        """Средний F1 поиска источников (0–1)."""
        return _mean(result.source_overlap.f1 for result in self.results)

    @property
    def mean_source_full_hit(self) -> float:
        """Доля кейсов, где нашлись все ожидаемые темы (0–1)."""
        return _mean(result.source_overlap.full_hit for result in self.results)

    @property
    def mean_source_mrr(self) -> float:
        """Среднее обратное место первой найденной темы (0–1)."""
        return _mean(result.source_overlap.mrr for result in self.results)

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
                "failed": self.failed_count,
                "judge": {
                    "accuracy": round(self.mean_accuracy, 4),
                    "completeness": round(self.mean_completeness, 4),
                    "relevance": round(self.mean_relevance, 4),
                },
                "lexical_overlap": {
                    "precision": round(self.mean_lexical_precision, 4),
                    "recall": round(self.mean_lexical_recall, 4),
                    "f1": round(self.mean_lexical_f1, 4),
                },
                "semantic_similarity": round(self.mean_semantic_similarity, 4),
                "source_overlap": {
                    "precision": round(self.mean_source_precision, 4),
                    "recall": round(self.mean_source_recall, 4),
                    "f1": round(self.mean_source_f1, 4),
                    "full_hit": round(self.mean_source_full_hit, 4),
                    "mrr": round(self.mean_source_mrr, 4),
                },
                "timing": {
                    "time_to_first_token": round(self.mean_time_to_first_token, 4),
                    "total": round(self.mean_total_time, 4),
                },
            },
            "cases": [self._case_to_dict(result) for result in self.results],
            "failures": [self._failure_to_dict(failure) for failure in self.failures],
        }

    def render_text(self) -> str:
        """Отчёт для чтения человеком в терминале."""
        lines = [
            "",
            f"Оценка терминала: {self.count} кейс(ов), сбоев: {self.failed_count}",
            "",
            "Средние оценки судьи (1–5)",
            f"  Точность      : {self.mean_accuracy:.2f}",
            f"  Полнота       : {self.mean_completeness:.2f}",
            f"  Релевантность : {self.mean_relevance:.2f}",
            "",
            "Близость ответа к эталону",
            f"  Лексическая (значимые слова) : F1 {self.mean_lexical_f1:.2f}",
            f"  Семантическая (эмбеддинги)   : {self.mean_semantic_similarity:.2f}",
            "",
            "Поиск источников",
            f"  Полнота (recall)   : {self.mean_source_recall:.2f}",
            f"  Все темы найдены   : {self.mean_source_full_hit:.2f}",
            f"  MRR (место темы)   : {self.mean_source_mrr:.2f}",
            f"  Точность (precision, ограничена бюджетом) : {self.mean_source_precision:.2f}",
            "",
            "Время ответа",
            f"  До первого токена : {self.mean_time_to_first_token:.3f} с",
            f"  Полное время      : {self.mean_total_time:.3f} с",
            "",
            "По кейсам",
            "  #    вопрос                          точн полн рел  лекс  сем истR MRR TTFT  время",
        ]
        for position, result in enumerate(self.results, start=1):
            lines.append(
                f"  {position:<4} {_shorten(result.case.question):<32} "
                f"{result.judge.accuracy:>4.0f} {result.judge.completeness:>4.0f} "
                f"{result.judge.relevance:>3.0f} "
                f"{result.lexical_overlap.f1:>5.2f} {result.semantic_similarity:>4.2f} "
                f"{result.source_overlap.recall:>4.2f} {result.source_overlap.mrr:>4.2f} "
                f"{result.timing.time_to_first_token:>5.2f}с {result.timing.total:>5.2f}с"
            )
        lines.extend(self._render_failures())
        lines.append("")
        return "\n".join(lines)

    def _render_failures(self) -> list[str]:
        """Список кейсов, оставшихся без ответа, — отдельно от средних."""
        if not self.failures:
            return []
        lines = ["", "Сбои (в средние не вошли)", "  #    вопрос                          причина"]
        for position, failure in enumerate(self.failures, start=1):
            lines.append(
                f"  {position:<4} {_shorten(failure.case.question):<32} {failure.reason}"
            )
        return lines

    @staticmethod
    def _case_to_dict(result: CaseResult) -> dict[str, object]:
        """Один кейс в виде словаря — для записи в JSON."""
        return {
            "question": result.case.question,
            "character_id": result.case.character_id,
            "tags": list(result.case.tags),
            "answer": result.answer,
            "sources": list(result.sources),
            "judge": {
                "accuracy": result.judge.accuracy,
                "completeness": result.judge.completeness,
                "relevance": result.judge.relevance,
                "comment": result.judge.comment,
            },
            "lexical_overlap": {
                "precision": round(result.lexical_overlap.precision, 4),
                "recall": round(result.lexical_overlap.recall, 4),
                "f1": round(result.lexical_overlap.f1, 4),
            },
            "semantic_similarity": round(result.semantic_similarity, 4),
            "source_overlap": {
                "precision": round(result.source_overlap.precision, 4),
                "recall": round(result.source_overlap.recall, 4),
                "f1": round(result.source_overlap.f1, 4),
                "full_hit": result.source_overlap.full_hit,
                "mrr": round(result.source_overlap.mrr, 4),
            },
            "timing": {
                "time_to_first_token": round(result.timing.time_to_first_token, 4),
                "total": round(result.timing.total, 4),
            },
        }

    @staticmethod
    def _failure_to_dict(failure: CaseFailure) -> dict[str, object]:
        """Один сбой в виде словаря — для записи в JSON."""
        return {
            "question": failure.case.question,
            "character_id": failure.case.character_id,
            "tags": list(failure.case.tags),
            "reason": failure.reason,
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
