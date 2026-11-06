from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from app.evaluation.domain import Overlap, SourceOverlap

WORD_PATTERN = re.compile(r"[0-9a-zа-яё]+")
"""Слово — это последовательность букв и цифр; пунктуация и регистр отброшены."""

STOP_WORDS: frozenset[str] = frozenset(
    (
        "а", "без", "бы", "был", "была", "были", "было", "быть", "в", "вам", "вас",
        "вдруг", "ведь", "во", "вот", "все", "всего", "всех", "вы", "где", "да",
        "даже", "для", "до", "его", "ее", "ей", "ему", "если", "есть", "еще", "ж",
        "же", "за", "здесь", "и", "из", "или", "им", "их", "к", "как", "кто", "ли",
        "либо", "мне", "может", "мои", "мой", "мы", "на", "надо", "нам", "нас",
        "не", "него", "нее", "ней", "нет", "ни", "них", "но", "ну", "о", "об",
        "один", "он", "она", "они", "опять", "от", "по", "под", "после", "потом",
        "потому", "почти", "при", "про", "раз", "с", "сам", "свой", "себе", "себя",
        "сейчас", "со", "совсем", "так", "такой", "там", "тебя", "тем", "теперь",
        "то", "тогда", "того", "тоже", "той", "только", "том", "тот", "ту", "тут",
        "ты", "у", "уж", "уже", "чем", "через", "что", "чтоб", "чтобы", "эта",
        "эти", "это", "этой", "этом", "этот", "эту", "я",
    )
)
"""Служебные слова: в пересечении текстов они дают совпадение на пустом месте."""

MIN_STEM: int = 4
"""Короче этого остатка слово не усекается: срезав больше, разные слова сойдутся."""

_SUFFIXES: tuple[str, ...] = tuple(
    sorted(
        (
            "иями", "ями", "ами", "ыми", "ими", "его", "ого", "ему", "ому",
            "ешь", "ишь", "ете", "ите", "ать", "ять", "еть", "ить", "ует", "уют",
            "ется", "ются", "ться", "лся", "лась", "лось", "лись", "ших", "ший",
            "шая", "шее", "шие", "щую", "ющий", "ющим",
            "ой", "ей", "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий", "ым", "им",
            "ом", "ем", "ах", "ях", "ов", "ев", "ам", "ям", "ую", "юю", "ия", "ию",
            "ьи", "а", "я", "о", "е", "у", "ю", "ы", "и", "ь",
        ),
        key=len,
        reverse=True,
    )
)


def tokenize(text: str) -> list[str]:
    """Разбить текст на слова в нижнем регистре."""
    return WORD_PATTERN.findall(text.lower())


def stem(word: str) -> str:
    """Отрезать у слова изменяемое окончание, оставив основу.

    Русский язык флективный: «танк», «танка» и «танку» — одно слово, но три
    разных строки. Без этого шага пересечение текстов занижено на ровном месте,
    потому что один и тот же смысл оформлен разными окончаниями.
    """
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM:
            return word[: -len(suffix)]
    return word


def content_tokens(text: str) -> list[str]:
    """Оставить от текста только значимые слова в основной форме.

    Служебные слова выбрасываются, остальные приводятся к основе: так пересечение
    становится мерой общего смысла, а не совпадения местоимений.
    """
    return [stem(word) for word in tokenize(text) if word not in STOP_WORDS]


def text_overlap(reference: str, candidate: str) -> Overlap:
    """Сравнить ответ модели с эталоном по значимым словам в основной форме.

    precision — какая доля слов ответа встречается в эталоне;
    recall — какую долю значимых слов эталона покрыл ответ;
    f1 — гармоническое среднее этих двух чисел, один балл вместо двух.

    Метрика поверхностная: она ловит общую лексику, но не пересказ своими словами.
    Рядом с ней считается семантическая близость, которая пересказ видит.
    """
    reference_counts = Counter(content_tokens(reference))
    candidate_counts = Counter(content_tokens(candidate))
    common = sum((reference_counts & candidate_counts).values())

    precision = _ratio(common, sum(candidate_counts.values()))
    recall = _ratio(common, sum(reference_counts.values()))
    return Overlap(precision=precision, recall=recall, f1=_f1(precision, recall))


def source_overlap(expected: Iterable[str], found: Iterable[str]) -> SourceOverlap:
    """Сравнить ожидаемые темы с заголовками найденных источников.

    Сравнение нестрогое, в обе стороны: ожидаемая тема «Танк» и заголовок
    «Танк › Базовые стратегии» считаются совпадением, потому что одна строка
    содержится в другой. Заголовки архива иерархические, и точное равенство строк
    не нашло бы ничего.

    Повторяющиеся заголовки схлопываются: бюджет выборки фиксирован, и одна и та
    же статья, пришедшая несколькими фрагментами, не должна выглядеть как
    несколько попаданий. ``precision`` всё равно ограничен бюджетом, поэтому о
    качестве ранжирования говорят ``full_hit`` и ``mrr``.
    """
    expected_items = _dedupe(_normalise_all(expected))
    found_items = _dedupe(_normalise_all(found))

    covered_expected = _count_covered(expected_items, found_items)
    covered_found = _count_covered(found_items, expected_items)

    precision = _ratio(covered_found, len(found_items))
    recall = _ratio(covered_expected, len(expected_items))
    return SourceOverlap(
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        full_hit=1.0 if expected_items and covered_expected == len(expected_items) else 0.0,
        mrr=_reciprocal_rank(expected_items, found_items),
    )


def _normalise_all(items: Iterable[str]) -> list[str]:
    """Привести элементы к виду «слова через пробел» и выбросить пустые."""
    normalised = [" ".join(tokenize(item)) for item in items]
    return [item for item in normalised if item]


def _dedupe(items: list[str]) -> list[str]:
    """Убрать повторы, сохранив порядок первого появления."""
    return list(dict.fromkeys(items))


def _count_covered(items: list[str], pool: list[str]) -> int:
    """Сколько элементов ``items`` совпало хотя бы с одним из ``pool``.

    Совпадением считается вхождение одной строки в другую — этого достаточно,
    чтобы тема «Танк» нашлась в заголовке «Танк › Базовые стратегии».
    """
    return sum(1 for item in items if any(item in other or other in item for other in pool))


def _reciprocal_rank(expected: list[str], found: list[str]) -> float:
    """Обратное место первой найденной ожидаемой темы; ноль, если её нет."""
    if not expected:
        return 0.0
    for rank, item in enumerate(found, start=1):
        if any(item in other or other in item for other in expected):
            return 1.0 / rank
    return 0.0


def _ratio(part: int, whole: int) -> float:
    """Доля ``part`` от ``whole``; ноль, если считать не от чего."""
    return part / whole if whole else 0.0


def _f1(precision: float, recall: float) -> float:
    """Гармоническое среднее точности и полноты."""
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
