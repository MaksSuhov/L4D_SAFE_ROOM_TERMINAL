"""Семантическая близость ответа и эталона — в дополнение к совпавшим словам.

Ответы терминала намеренно пересказывают эталон своими словами и в образе
персонажа, поэтому лексическое пересечение у них низкое по построению. Косинус
двух эмбеддингов ловит то, чего не видит пересечение слов: пересказ остаётся
близким даже тогда, когда общих слов почти нет.
"""

from __future__ import annotations

import numpy as np

from app.evaluation.interfaces import EmbeddingProtocol


class NullSimilarity:
    """Заглушка для прогонов без модели эмбеддингов: близость не считается."""

    async def similarity(self, reference: str, answer: str) -> float:
        """Вернуть ноль: измерять нечем."""
        return 0.0


class EmbeddingSimilarity:
    """Косинусная близость двух текстов в пространстве эмбеддингов модели."""

    def __init__(self, embeddings: EmbeddingProtocol) -> None:
        self._embeddings = embeddings

    async def similarity(self, reference: str, answer: str) -> float:
        """Посчитать близость ``answer`` к ``reference`` одним пакетным запросом.

        Пустой эталон или пустой ответ близости не имеют: возвращается ноль, и
        запрос к провайдеру не уходит.
        """
        if not reference.strip() or not answer.strip():
            return 0.0
        reference_vector, answer_vector = await self._embeddings.get_embeddings(
            [reference, answer]
        )
        return _cosine(reference_vector, answer_vector)


def _cosine(left: list[float], right: list[float]) -> float:
    """Косинус двух векторов, обрезанный снизу нулём.

    Отрицательный косинус означает противоположные направления, а не
    «отрицательно похожие» тексты; для отчёта он приводится к нулю.
    """
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.clip(a @ b / denominator, 0.0, 1.0))
