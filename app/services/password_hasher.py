from __future__ import annotations

import secrets
from typing import Final

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

DEFAULT_MEMORY_COST_KIB: Final[int] = 65536
"""Память на один расчёт хеша: 64 МиБ делают перебор по словарю дорогим."""

DEFAULT_TIME_COST: Final[int] = 3
"""Число проходов: вместе с памятью задаёт стоимость одной проверки пароля."""

DEFAULT_PARALLELISM: Final[int] = 4
"""Сколько потоков вправе занять один расчёт — ровно столько ядер и ожидается."""


class Argon2PasswordHasher:
    """Хеширование паролей алгоритмом Argon2id.

    Соль и параметры алгоритма зашиты в саму строку хеша, поэтому смена
    параметров не ломает сохранённые пароли: старый хеш продолжает проверяться, а
    :meth:`needs_rehash` подсказывает, что при следующем входе его пора пересчитать.
    """

    def __init__(
        self,
        *,
        memory_cost_kib: int = DEFAULT_MEMORY_COST_KIB,
        time_cost: int = DEFAULT_TIME_COST,
        parallelism: int = DEFAULT_PARALLELISM,
    ) -> None:
        self._hasher = PasswordHasher(
            memory_cost=memory_cost_kib,
            time_cost=time_cost,
            parallelism=parallelism,
        )
        # Строка-заглушка считается теми же параметрами, что и настоящие хеши:
        # иначе проверка «такой учётной записи нет» шла бы не то же время, что
        # проверка неверного пароля, и разница выдала бы наличие записи.
        self._timing_equaliser = self._hasher.hash(secrets.token_urlsafe(16))

    def hash(self, password: str) -> str:
        """Посчитать хеш пароля вместе с солью и параметрами алгоритма."""
        return self._hasher.hash(password)

    def verify(self, password: str, encoded: str) -> bool:
        """Проверить пароль против ранее сохранённого хеша."""
        try:
            return self._hasher.verify(encoded, password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False

    def needs_rehash(self, encoded: str) -> bool:
        """Правда ли, что хеш посчитан с устаревшими параметрами."""
        try:
            return self._hasher.check_needs_rehash(encoded)
        except InvalidHashError:
            return True

    def equalise_timing(self, password: str) -> None:
        """Проделать работу проверки, ничего не проверяя."""
        self.verify(password, self._timing_equaliser)
