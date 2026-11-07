from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class RefreshSession:
    """Выданный клиенту refresh-токен, каким его видит сервер.

    Серверной копии токена не существует: хранится только его отпечаток. Это
    делает сессии отзываемыми — их видно, их можно погасить по цепочке ротаций, —
    и при этом бесполезными для того, кто прочитает хранилище.
    """

    user_id: str
    token_digest: str
    """Отпечаток самого токена; сырое значение не хранится нигде."""
    family_id: str
    """Идентификатор цепочки ротаций: повторное использование гасит всю цепочку."""
    issued_at: datetime
    expires_at: datetime
    id: str | None = None
    """Идентификатор записи; до сохранения его нет, его выдаёт хранилище."""
    revoked_at: datetime | None = None
    replaced_by: str | None = None
    """Сессия, выданная взамен этой при ротации: след цепочки для разбора инцидентов."""
    user_agent: str = ""
    """Клиент, которому выдан токен; нужен для разбора, а не для проверки прав."""

    @property
    def is_revoked(self) -> bool:
        """Правда ли, что сессия погашена — выходом, ротацией или тревогой."""
        return self.revoked_at is not None

    @property
    def identifier(self) -> str:
        """Идентификатор сохранённой сессии."""
        if self.id is None:
            raise ValueError("the session has not been stored yet and has no identifier")
        return self.id

    def is_expired(self, now: datetime) -> bool:
        """Правда ли, что срок жизни сессии истёк к моменту ``now``."""
        return now >= self.expires_at

    def is_usable(self, now: datetime) -> bool:
        """Правда ли, что по этой сессии ещё можно выдать новую пару токенов."""
        return not self.is_revoked and not self.is_expired(now)
