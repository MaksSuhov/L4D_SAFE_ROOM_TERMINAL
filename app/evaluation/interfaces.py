"""Порты модуля оценки.

Каждый шаг зависит от узкого протокола, а не от конкретного клиента: судье
нужен только текстовый ответ модели, замеру — часы, прогону — сборщик
контекста. Поэтому в тесте можно подменить ровно одну деталь, а не тащить за
собой провайдера и базу.

Конкретные классы (``OpenAIClient``, ``RAGService``) подходят под эти протоколы
по своей форме — наследоваться от них не нужно.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Protocol

from app.evaluation.domain import JudgeScore
from app.interfaces import PreparedChat


class ClockProtocol(Protocol):
    """Источник монотонного времени в секундах."""

    def now(self) -> float:
        """Вернуть текущее время."""
        ...


class CompletionProtocol(Protocol):
    """Один текстовый ответ модели на одну инструкцию."""

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.0,
    ) -> str:
        """Ответить на ``user_message`` согласно ``system_prompt``."""
        ...


class StreamingChatProtocol(Protocol):
    """Генерация ответа в образе персонажа, отдаваемая по кускам."""

    model_name: str

    def stream_response(
        self,
        system_prompt: str,
        context: str,
        user_message: str,
        few_shot_examples: Sequence[Mapping[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """Отдавать ответ по фрагментам по мере их появления."""
        ...


class ContextBuilderProtocol(Protocol):
    """Поиск по архиву: вопрос превращается в готовый контекст и источники."""

    async def build_chat_context(self, character_id: int, user_message: str) -> PreparedChat:
        """Собрать контекст одного хода разговора."""
        ...


class JudgeProtocol(Protocol):
    """Оценка одного ответа по трём критериям."""

    async def evaluate(
        self,
        *,
        question: str,
        answer: str,
        reference: str = "",
        context: str = "",
    ) -> JudgeScore:
        """Оценить ``answer`` и вернуть баллы с объяснением."""
        ...
