"""Проверки хеширования паролей: соль, проверка, пересчёт и выравнивание времени."""

from __future__ import annotations

from app.services import Argon2PasswordHasher

CHEAP: dict[str, int] = {"memory_cost_kib": 8192, "time_cost": 1, "parallelism": 1}
"""Параметры подешевле: тест проверяет логику, а не стоимость перебора."""


def test_the_same_password_hashes_differently_every_time() -> None:
    """Соль обязана быть случайной: иначе одинаковые пароли видно по одинаковым хешам."""
    hasher = Argon2PasswordHasher(**CHEAP)

    first = hasher.hash("correct-horse-battery")
    second = hasher.hash("correct-horse-battery")

    assert first != second
    assert hasher.verify("correct-horse-battery", first)
    assert hasher.verify("correct-horse-battery", second)


def test_a_wrong_password_does_not_verify() -> None:
    hasher = Argon2PasswordHasher(**CHEAP)
    encoded = hasher.hash("correct-horse-battery")

    assert hasher.verify("Correct-Horse-Battery", encoded) is False
    assert hasher.verify("", encoded) is False


def test_a_foreign_string_is_a_failed_check_not_an_error() -> None:
    """Чужой хеш — это «не подходит», а не исключение: так отвечает и база после переезда."""
    hasher = Argon2PasswordHasher(**CHEAP)

    assert hasher.verify("correct-horse-battery", "not-a-hash") is False
    assert hasher.needs_rehash("not-a-hash") is True


def test_hashes_with_changed_parameters_ask_to_be_recomputed() -> None:
    """Смена параметров алгоритма не ломает старый хеш, но помечает его устаревшим."""
    old = Argon2PasswordHasher(**CHEAP)
    new = Argon2PasswordHasher(**{**CHEAP, "time_cost": 2})

    encoded = old.hash("correct-horse-battery")

    assert old.needs_rehash(encoded) is False
    assert new.needs_rehash(encoded) is True
    assert new.verify("correct-horse-battery", encoded) is True


def test_timing_equalisation_accepts_any_password() -> None:
    """Выравнивание времени обязано молча отработать на любом пароле."""
    hasher = Argon2PasswordHasher(**CHEAP)

    hasher.equalise_timing("whatever-the-visitor-typed")
