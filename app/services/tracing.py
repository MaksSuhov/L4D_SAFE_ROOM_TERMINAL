"""Адаптеры трейсинга: настоящий (Langfuse) и выключенный.

Сервисы знают только порт :class:`~app.interfaces.TracerProtocol`; единственное
место, которое знает про SDK наблюдаемости, — этот модуль. Выключенный трейсер не
стоит ничего, поэтому тесты и локальный запуск без ключей идут ровно тем же кодом,
что и продакшен.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from types import TracebackType
from typing import Any

from langfuse import Langfuse

from app.config import Settings
from app.interfaces import TracerProtocol, TracingSpanProtocol


class NoOpTracingSpan:
    """Участок выключенного трейсера: принимает разметку и ничего с ней не делает."""

    def annotate(
        self,
        *,
        input: Any = None,
        output: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Принять разметку участка и отбросить её."""


class NoOpTracer:
    """Выключенный трейсер: конвейер работает ровно так, как без наблюдаемости."""

    def span(
        self,
        name: str,
        *,
        as_type: str = "span",
    ) -> AbstractContextManager[TracingSpanProtocol]:
        """Открыть участок, который никуда не попадёт."""
        return nullcontext(NoOpTracingSpan())

    def flush(self) -> None:
        """Нечего отправлять."""

    def shutdown(self) -> None:
        """Нечего освобождать."""


class _LangfuseSpan:
    """Наблюдение Langfuse, приведённое к порту приложения.

    Контекстный менеджер отдаёт само наблюдение только на входе в блок, поэтому
    разметка ждёт ``__enter__``: до него обновлять ещё нечего.
    """

    def __init__(self, manager: AbstractContextManager[Any]) -> None:
        self._manager = manager
        self._observation: Any | None = None

    def __enter__(self) -> _LangfuseSpan:
        self._observation = self._manager.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._manager.__exit__(exc_type, exc_value, traceback)

    def annotate(
        self,
        *,
        input: Any = None,
        output: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Передать разметку участка в Langfuse, пропустив незаполненные поля."""
        fields = {
            name: value
            for name, value in (
                ("input", input),
                ("output", output),
                ("metadata", metadata),
            )
            if value is not None
        }
        if fields and self._observation is not None:
            self._observation.update(**fields)


class LangfuseTracer:
    """Трейсер поверх клиента Langfuse.

    Клиент строится здесь, а не берётся из окружения: только явно созданный клиент
    находит drop-in ``langfuse.openai``, которым трассируются сами вызовы модели.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
        )

    def span(
        self,
        name: str,
        *,
        as_type: str = "span",
    ) -> AbstractContextManager[TracingSpanProtocol]:
        """Открыть наблюдение ``name`` в текущем трейсе."""
        # The SDK types the kind as a Literal of its own vocabulary, and the port
        # deliberately keeps a plain string so that the service layer never imports
        # SDK types; the widening is what the adapter is for.
        kind: Any = as_type
        return _LangfuseSpan(
            self._client.start_as_current_observation(name=name, as_type=kind)
        )

    def flush(self) -> None:
        """Дождаться отправки накопленных наблюдений."""
        self._client.flush()

    def shutdown(self) -> None:
        """Остановить фоновые очереди и освободить клиент."""
        self._client.shutdown()


def build_tracer(settings: Settings) -> TracerProtocol:
    """Собрать трейсер по настройкам: Langfuse при ключах, иначе выключенный.

    Ключи решают, а не только выключатель: без пары ключей клиенту нечего
    отправлять, и трейсер обязан быть no-op, чтобы не тратить вызовы модели на
    разметку, которую никто не увидит.
    """
    if not (settings.langfuse_enabled and settings.is_langfuse_configured):
        return NoOpTracer()
    return LangfuseTracer(settings)


__all__ = ["LangfuseTracer", "NoOpTracer", "NoOpTracingSpan", "build_tracer"]
