from __future__ import annotations

from datetime import UTC, datetime


class SystemClock:
    """Часы процесса: текущий момент в UTC и с указанием часового пояса."""

    def now(self) -> datetime:
        """Вернуть текущий момент в UTC."""
        return datetime.now(UTC)
