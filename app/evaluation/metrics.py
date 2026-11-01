"""Метрики пересечения текста и множеств.

Здесь только чистые функции: на вход строки, на выход числа. Их одинаково
используют и отчёт, и тесты, и они ничего не знают про модель.

Метрики две:

* :func:`text_overlap` — сравнивает ответ модели с эталонным ответом по словам.
  Это привычные precision, recall и F1 из оценки качества ответов: сколько
  сказанного есть в эталоне, сколько эталона покрыто, и среднее из двух.
* :func:`set_overlap` — сравнивает ожидаемые темы с заголовками найденных
  источников. Отвечает на вопрос «нашёл ли поиск то, что должен был».
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from app.evaluation.domain import Overlap

WORD_PATTERN = re.compile(r"[0-9a-zа-яё]+")
"""Слово — это последовательность букв и цифр; пунктуация и регистр отброшены."""


def tokenize(text: str) -> list[str]:
    """Разбить текст на слова в нижнем регистре."""
    return WORD_PATTERN.findall(text.lower())


def text_overlap(reference: str, candidate: str) -> Overlap:
    """Сравнить ответ модели с эталоном по совпавшим словам.

    precision — какая доля слов ответа встречается в эталоне;
    recall — какую долю слов эталона покрыл ответ;
    f1 — гармоническое среднее этих двух чисел, один балл вместо двух.
    """
    reference_counts = Counter(tokenize(reference))
    candidate_counts = Counter(tokenize(candidate))
    common = sum((reference_counts & candidate_counts).values())

    precision = _ratio(common, sum(candidate_counts.values()))
    recall = _ratio(common, sum(reference_counts.values()))
    return Overlap(precision=precision, recall=recall, f1=_f1(precision, recall))


def set_overlap(expected: Iterable[str], found: Iterable[str]) -> Overlap:
    """Сравнить ожидаемые темы с заголовками найденных источников.

    Сравнение нестрогое, в обе стороны: ожидаемая тема «Танк» и заголовок
    «Танк › Базовые стратегии» считаются совпадением, потому что одна строка
    содержится в другой. Заголовки архива иерархические, и точное равенство
    строк не нашло бы ничего.
    """
    expected_items = _normalise_all(expected)
    found_items = _normalise_all(found)

    covered_expected = _count_covered(expected_items, found_items)
    covered_found = _count_covered(found_items, expected_items)

    precision = _ratio(covered_found, len(found_items))
    recall = _ratio(covered_expected, len(expected_items))
    return Overlap(precision=precision, recall=recall, f1=_f1(precision, recall))


def _normalise_all(items: Iterable[str]) -> list[str]:
    """Привести элементы к виду «слова через пробел» и выбросить пустые."""
    normalised = [" ".join(tokenize(item)) for item in items]
    return [item for item in normalised if item]


def _count_covered(items: list[str], pool: list[str]) -> int:
    """Сколько элементов ``items`` совпало хотя бы с одним из ``pool``.

    Совпадением считается вхождение одной строки в другую — этого достаточно,
    чтобы тема «Танк» нашлась в заголовке «Танк › Базовые стратегии».
    """
    return sum(1 for item in items if any(item in other or other in item for other in pool))


def _ratio(part: int, whole: int) -> float:
    """Доля ``part`` от ``whole``; ноль, если считать не от чего."""
    return part / whole if whole else 0.0


def _f1(precision: float, recall: float) -> float:
    """Гармоническое среднее точности и полноты."""
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
