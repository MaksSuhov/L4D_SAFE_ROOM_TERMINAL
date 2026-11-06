from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Protocol

from app.evaluation.domain import JudgeScore
from app.interfaces import PreparedChat


class ClockProtocol(Protocol):

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
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Отдавать ответ по фрагментам по мере их появления.

        ``temperature`` переопределяет настройку на один вызов: стенд закрепляет
        её, чтобы прогоны были сравнимы.
        """
        ...


class EmbeddingProtocol(Protocol):
    """Эмбеддинги текстов — узкий срез возможностей клиента модели.

    Пакету оценки нужна ровно эта операция, поэтому порт объявляет её одну, а не
    тянет за собой весь :class:`~app.interfaces.AIClientProtocol`.
    """

    async def get_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """Посчитать эмбеддинги пакета текстов, сохранив их порядок."""
        ...


class SimilarityProtocol(Protocol):
    """Близость ответа к эталону одним числом от 0 до 1."""

    async def similarity(self, reference: str, answer: str) -> float:
        """Вернуть, насколько ``answer`` близок к ``reference``."""
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
