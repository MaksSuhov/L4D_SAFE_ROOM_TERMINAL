"""Тесты чистых метрик пересечения: ни модели, ни базы, ни сети."""

from __future__ import annotations

import pytest

from app.evaluation.metrics import content_tokens, source_overlap, stem, text_overlap, tokenize


class TestTokenize:
    def test_lowercases_and_drops_punctuation(self) -> None:
        assert tokenize("Ведьма, ушла! 42.") == ["ведьма", "ушла", "42"]

    def test_keeps_cyrillic_and_latin(self) -> None:
        assert tokenize("Tank — Танк") == ["tank", "танк"]

    def test_blank_text_has_no_tokens(self) -> None:
        assert tokenize("   ...   ") == []


class TestStem:
    def test_inflected_forms_share_a_stem(self) -> None:
        assert stem("танка") == stem("танку") == stem("танк")

    def test_a_short_word_is_left_alone(self) -> None:
        """Проверить, что короткое слово не усекается до пустоты."""
        assert stem("яд") == "яд"


class TestContentTokens:
    def test_drops_stop_words(self) -> None:
        assert content_tokens("Танк и Ведьма") == ["танк", "ведьм"]

    def test_brings_words_to_their_stem(self) -> None:
        assert content_tokens("выжившие выживших") == ["выжив", "выжив"]


class TestTextOverlap:
    def test_identical_texts_score_perfectly(self) -> None:
        overlap = text_overlap("стены за спиной", "стены за спиной")

        assert overlap.precision == 1.0
        assert overlap.recall == 1.0
        assert overlap.f1 == 1.0

    def test_precision_and_recall_split_when_text_differs(self) -> None:
        """Проверить, что precision и recall расходятся: в ответе два слова из эталона и
        одно лишнее.
        """
        overlap = text_overlap("держи дистанцию", "держи дистанцию бегом")

        assert overlap.precision == pytest.approx(2 / 3)
        assert overlap.recall == 1.0
        assert overlap.f1 == pytest.approx(0.8)

    def test_inflected_words_still_match(self) -> None:
        """Проверить, что разные окончания одного слова не считаются разными словами."""
        overlap = text_overlap("Танк швыряет камни", "камни швырял танк")

        assert overlap.recall == pytest.approx(2 / 3)
        assert overlap.precision == pytest.approx(2 / 3)

    def test_empty_reference_leaves_recall_at_zero(self) -> None:
        overlap = text_overlap("", "любой ответ")

        assert overlap.precision == 0.0
        assert overlap.recall == 0.0
        assert overlap.f1 == 0.0

    def test_repeated_words_are_counted_once_per_occurrence(self) -> None:
        overlap = text_overlap("танк", "танк танк")

        assert overlap.precision == pytest.approx(0.5)
        assert overlap.recall == 1.0


class TestSourceOverlap:
    def test_expected_topic_inside_a_hierarchical_title_counts_as_found(self) -> None:
        overlap = source_overlap(["Танк"], ["Танк › Базовые стратегии"])

        assert overlap.precision == 1.0
        assert overlap.recall == 1.0
        assert overlap.f1 == 1.0
        assert overlap.full_hit == 1.0
        assert overlap.mrr == 1.0

    def test_missing_topic_lowers_recall(self) -> None:
        overlap = source_overlap(["Танк", "Ведьма"], ["Танк › Обзор"])

        assert overlap.precision == 1.0
        assert overlap.recall == pytest.approx(0.5)
        assert overlap.full_hit == 0.0

    def test_irrelevant_source_lowers_precision(self) -> None:
        overlap = source_overlap(["Танк"], ["Танк › Обзор", "Курильщик › Обзор"])

        assert overlap.precision == pytest.approx(0.5)
        assert overlap.recall == 1.0

    def test_repeated_titles_are_counted_once(self) -> None:
        """Проверить, что статья, пришедшая несколькими фрагментами, — одно попадание."""
        overlap = source_overlap(["Танк"], ["Танк › Разное", "Танк › Разное", "Жокей › Обзор"])

        assert overlap.precision == pytest.approx(0.5)
        assert overlap.recall == 1.0

    def test_mrr_reflects_the_rank_of_the_first_found_topic(self) -> None:
        overlap = source_overlap(["Танк"], ["Жокей › Обзор", "Танк › Обзор"])

        assert overlap.mrr == pytest.approx(0.5)
        assert overlap.recall == 1.0

    def test_no_expected_topics_mean_no_measurement(self) -> None:
        overlap = source_overlap([], ["Танк › Обзор"])

        assert overlap.precision == 0.0
        assert overlap.recall == 0.0
        assert overlap.f1 == 0.0
        assert overlap.full_hit == 0.0
        assert overlap.mrr == 0.0
