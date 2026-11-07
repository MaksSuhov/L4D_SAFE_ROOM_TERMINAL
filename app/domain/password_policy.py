from __future__ import annotations

from typing import Final

from app.exceptions import WeakPasswordError

MIN_PASSWORD_LENGTH: Final[int] = 10
"""Нижняя граница длины пароля: та же константа стоит в схеме запроса регистрации."""

MAX_PASSWORD_LENGTH: Final[int] = 128
"""Верхняя граница: хеширование длинного пароля — работа, которой вправе не быть."""

COMMON_PASSWORDS: Final[frozenset[str]] = frozenset(
    {
        "1234567890",
        "qwertyuiop",
        "password123",
        "пароль12345",
        "l4dterminal",
        "safe-room",
        "survivor123",
        "administrator",
        "letmein12345",
        "iloveyou123",
    }
)
"""Короткий список самых ходовых паролей; полный список встречаемости проекту ни к чему."""


class PasswordPolicy:
    """Правило, которому обязан отвечать пароль учётной записи.

    Состав символов не проверяется намеренно: сложность набора требований толкает
    людей к предсказуемым заменам («Пароль1!»), тогда как длина и отсечение
    заведомо угнанных паролей дают больше. Так же смотрит на это NIST SP 800-63B.
    """

    def __init__(
        self,
        *,
        min_length: int = MIN_PASSWORD_LENGTH,
        max_length: int = MAX_PASSWORD_LENGTH,
        blocked: frozenset[str] = COMMON_PASSWORDS,
    ) -> None:
        self._min_length = min_length
        self._max_length = max_length
        self._blocked = blocked

    @property
    def min_length(self) -> int:
        """Нижняя граница длины, как её объявляет политика."""
        return self._min_length

    def validate(self, password: str) -> None:
        """Поднять :class:`WeakPasswordError`, если пароль не отвечает правилу."""
        if len(password) < self._min_length:
            raise WeakPasswordError(f"Пароль короче {self._min_length} символов.")
        if len(password) > self._max_length:
            raise WeakPasswordError(f"Пароль длиннее {self._max_length} символов.")
        if password.strip() != password:
            raise WeakPasswordError("Пароль не должен начинаться или заканчиваться пробелом.")
        if password.casefold() in self._blocked:
            raise WeakPasswordError("Такой пароль слишком часто встречается, выберите другой.")
