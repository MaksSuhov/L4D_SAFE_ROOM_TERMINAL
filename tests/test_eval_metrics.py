"""Tests for the pure overlap metrics: no model, no database, no network."""

from __future__ import annotations

import pytest

from app.evaluation.metrics import set_overlap, text_overlap, tokenize


class TestTokenize:
    def test_lowercases_and_drops_punctuation(self) -> None:
        assert tokenize("Ведьма, ушла! 42.") == ["ведьма", "ушла", "42"]

    def test_keeps_cyrillic_and_latin(self) -> None:
        assert tokenize("Tank — Танк") == ["tank", "танк"]

    def test_blank_text_has_no_tokens(self) -> None:
        assert tokenize("   ...   ") == []


class TestTextOverlap:
    def test_identical_texts_score_perfectly(self) -> None:
        overlap = text_overlap("стены за спиной", "стены за спиной")

        assert overlap.precision == 1.0
        assert overlap.recall == 1.0
        assert overlap.f1 == 1.0

    def test_precision_and_recall_split_when_text_differs(self) -> None:
        """Answer says two of the reference words and one word too many."""
        overlap = text_overlap("держи дистанцию", "держи дистанцию бегом")

        assert overlap.precision == pytest.approx(2 / 3)
        assert overlap.recall == 1.0
        assert overlap.f1 == pytest.approx(0.8)

    def test_empty_reference_leaves_recall_at_zero(self) -> None:
        overlap = text_overlap("", "любой ответ")

        assert overlap.precision == 0.0
        assert overlap.recall == 0.0
        assert overlap.f1 == 0.0

    def test_repeated_words_are_counted_once_per_occurrence(self) -> None:
        overlap = text_overlap("танк", "танк танк")

        assert overlap.precision == pytest.approx(0.5)
        assert overlap.recall == 1.0


class TestSetOverlap:
    def test_expected_topic_inside_a_hierarchical_title_counts_as_found(self) -> None:
        overlap = set_overlap(["Танк"], ["Танк › Базовые стратегии"])

        assert overlap.precision == 1.0
        assert overlap.recall == 1.0
        assert overlap.f1 == 1.0

    def test_missing_topic_lowers_recall(self) -> None:
        overlap = set_overlap(["Танк", "Ведьма"], ["Танк › Обзор"])

        assert overlap.precision == 1.0
        assert overlap.recall == pytest.approx(0.5)

    def test_irrelevant_source_lowers_precision(self) -> None:
        overlap = set_overlap(["Танк"], ["Танк › Обзор", "Курильщик › Обзор"])

        assert overlap.precision == pytest.approx(0.5)
        assert overlap.recall == 1.0

    def test_no_expected_topics_mean_no_measurement(self) -> None:
        overlap = set_overlap([], ["Танк › Обзор"])

        assert overlap.precision == 0.0
        assert overlap.recall == 0.0
        assert overlap.f1 == 0.0
