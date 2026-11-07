"""Доменные сущности приложения.

Фасад пакета реэкспортирует каждую сущность, поэтому остальной код сохраняет
единый стабильный путь импорта (``from app.domain import LoreChunk``), пока
определения живут по одному модулю на агрегат.

Сущности — обычные датаклассы, а не мапперы ORM: домен не знает ни о SQL, ни о
векторном хранилище. Именно поэтому хранилище можно заменить целиком, не трогая
ни одного сервиса.
"""

from app.domain.character import Character
from app.domain.lore_chunk import LoreChunk
from app.domain.password_policy import (
    COMMON_PASSWORDS,
    MAX_PASSWORD_LENGTH,
    MIN_PASSWORD_LENGTH,
    PasswordPolicy,
)
from app.domain.refresh_session import RefreshSession
from app.domain.user import User, normalise_email

__all__ = [
    "COMMON_PASSWORDS",
    "MAX_PASSWORD_LENGTH",
    "MIN_PASSWORD_LENGTH",
    "Character",
    "LoreChunk",
    "PasswordPolicy",
    "RefreshSession",
    "User",
    "normalise_email",
]
