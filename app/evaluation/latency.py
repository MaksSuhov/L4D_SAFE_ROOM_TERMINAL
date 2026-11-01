"""Замеры времени ответа.

Потоковая генерация даёт две разные цифры, и обе важны:

* **время до первого токена** — сколько игрок ждёт, прежде чем на экране
  появится первый символ. Это то, что человек ощущает как «скорость ответа».
* **полное время** — сколько идёт генерация целиком.

Часы здесь — внешняя зависимость (:class:`~app.evaluation.interfaces.ClockProtocol`),
поэтому в тесте их можно подменить и получить точные числа без ожидания.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from app.evaluation.domain import Timing
from app.evaluation.interfaces import ClockProtocol


class MonotonicClock:
    """Часы на монотонном таймере: их не сдвигают перевод системного времени."""

    def now(self) -> float:
        """Вернуть текущее значение таймера в секундах."""
        return time.perf_counter()


async def measure_stream(tokens: AsyncIterator[str], clock: ClockProtocol) -> tuple[str, Timing]:
    """Собрать ответ целиком и замерить паузу до первого токена и общее время."""
    start = clock.now()
    first_token_at: float | None = None
    fragments: list[str] = []

    async for fragment in tokens:
        if first_token_at is None:
            first_token_at = clock.now()
        fragments.append(fragment)

    finished = clock.now()
    first = first_token_at if first_token_at is not None else finished
    return "".join(fragments), Timing(time_to_first_token=first - start, total=finished - start)
