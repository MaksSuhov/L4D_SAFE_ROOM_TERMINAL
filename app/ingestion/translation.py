"""Нормализация языка корпуса перед индексацией.

Вики написана на английском, а терминал отвечает на русском, и межъязыковой
векторный поиск слабее одноязычного: при точечной проверке на этом самом корпусе
русский вопрос про «Зелёный грипп» набрал 0.53 против английской записи, но 0.74
против её русского аналога, со значительно большим отрывом от второго места.
Поиск всё равно ставил нужную запись первой в каждом случае, поэтому корпус
пригоден и как есть — но развёртывание, которому нужны более тесные отрывы,
может один раз нормализовать корпус здесь.

Этап **opt-in** (``--translate``), намеренно. Он стоит одного chat-завершения
на пакет записей, а на всём корпусе это медленный проход: по замерам на
настроенном провайдере шестнадцать коротких строк заняли ~21 с, а один длинный
абзац статьи — ~47 с и вернулся непереведённым. На ~2 300 записей это работа на
десятки минут — разумная как разовая для корпуса, который будет отгружен, и
неверная как скрытое значение по умолчанию для каждого локального наполнения.
Текст, который провайдер не смог перевести, сохраняет исходную форму, а не
отбрасывается, поэтому частичный проход ухудшает, а не портит архив.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.interfaces import AIClientProtocol
from app.schemas import LoreChunkCreate

logger = logging.getLogger("l4d.ingestion.translation")


@dataclass(frozen=True, slots=True)
class TranslationOutcome:
    """Результат перевода корпуса."""

    documents: list[LoreChunkCreate]
    translated: int
    unchanged: int

    @property
    def total(self) -> int:
        """Сколько записей было на входе."""
        return self.translated + self.unchanged


class CorpusTranslator:
    """Переписывает ``content`` каждой записи на целевой язык."""

    def __init__(self, ai_client: AIClientProtocol, *, target_language: str) -> None:
        self._ai_client = ai_client
        self._target_language = target_language

    @property
    def target_language(self) -> str:
        """Язык, к которому приводится каждая запись."""
        return self._target_language

    async def translate(self, documents: Sequence[LoreChunkCreate]) -> TranslationOutcome:
        """Перевести записи и сообщить, сколько реально было переписано."""
        if not documents:
            return TranslationOutcome(documents=[], translated=0, unchanged=0)

        originals = [document.content for document in documents]
        logger.info(
            "Translating %d corpus record(s) into %s",
            len(originals),
            self._target_language,
        )
        translations = await self._ai_client.translate_texts(originals)

        translated_documents: list[LoreChunkCreate] = []
        translated = 0
        for document, original, translation in zip(documents, originals, translations, strict=True):
            replacement = translation.strip()
            # An empty reply would silently erase a record; the source text is
            # always a valid fallback.
            if not replacement:
                replacement = original
            if replacement != original:
                translated += 1
            translated_documents.append(
                document.model_copy(update={"content": replacement})
            )
        return TranslationOutcome(
            documents=translated_documents,
            translated=translated,
            unchanged=len(documents) - translated,
        )
