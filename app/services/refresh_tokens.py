from __future__ import annotations

import hashlib
import secrets
from typing import Final

from app.interfaces import MintedRefreshToken

REFRESH_TOKEN_BYTES: Final[int] = 48
"""Энтропия refresh-токена: 48 случайных байт перебором не берутся."""


class SecureRefreshTokenFactory:
    """Refresh-токены: случайная строка клиенту, отпечаток — в хранилище.

    Токен не подписывается и не несёт данных: его единственная задача — быть
    неугадываемым. Проверяется он не подписью, а поиском своей сессии, поэтому
    утечка хранилища не даёт предъявить чужой токен — там лежат только отпечатки.
    """

    def mint(self) -> MintedRefreshToken:
        """Выдать новый токен вместе с отпечатком, пригодным для хранения."""
        value = secrets.token_urlsafe(REFRESH_TOKEN_BYTES)
        return MintedRefreshToken(value=value, digest=self.digest(value))

    def digest(self, token: str) -> str:
        """Посчитать отпечаток предъявленного токена."""
        return hashlib.sha256(token.encode("utf-8")).hexdigest()
