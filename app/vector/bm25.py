"""Локальный энкодер разреженного вектора BM25.

Модель считается в процессе и не обращается ни к провайдеру, ни к серверному
инференсу: fastembed кладёт в вектор частоты терминов, а IDF применяет Qdrant
(``Modifier.IDF``). Модель синхронная, поэтому вызовы уходят в рабочий поток,
чтобы не блокировать цикл событий.

Порт объявлен здесь, а не в ``app.interfaces``: потребитель у него ровно один —
адаптер векторного хранилища, и слой сервисов о существовании BM25 не знает.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from typing import Final, Protocol

from fastembed import SparseTextEmbedding
from qdrant_client import models

DEFAULT_BATCH_SIZE: Final[int] = 64
"""Сколько текстов уходит в один проход модели: у BM25 он дешёвый, но не бесплатный."""


class SparseEncoderProtocol(Protocol):
    """Превращает текст в разреженный вектор одного и того же пространства терминов."""

    async def encode_documents(self, texts: Sequence[str]) -> list[models.SparseVector]:
        """Посчитать векторы записей корпуса, сохранив порядок ``texts``."""
        ...

    async def encode_query(self, text: str) -> models.SparseVector:
        """Посчитать вектор поискового запроса."""
        ...


class Bm25Encoder:
    """Энкодер разреженного вектора на модели BM25 из fastembed.

    Модель берётся из кэша fastembed при первом обращении, а не при построении
    объекта: ``Qdrant/bm25`` — это конфигурация с частотными параметрами и
    списками стоп-слов, и её загрузка требует сети ровно один раз. Отложив её,
    приложение поднимается и отвечает на пробы готовности офлайн.
    """

    def __init__(
        self,
        model_name: str,
        *,
        language: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model_name = model_name
        # Язык задаёт токенизацию, стеммер и стоп-слова: русский корпус и русский
        # вопрос обязаны проходить через один и тот же словарь.
        self._language = language
        self._batch_size = batch_size
        self._model: SparseTextEmbedding | None = None
        self._load_lock = threading.Lock()

    async def encode_documents(self, texts: Sequence[str]) -> list[models.SparseVector]:
        """Посчитать векторы записей корпуса, сохранив порядок ``texts``."""
        if not texts:
            return []
        batch = list(texts)
        return await asyncio.to_thread(self._embed, batch)

    async def encode_query(self, text: str) -> models.SparseVector:
        """Посчитать вектор поискового запроса; IDF к нему добавит сервер."""
        (vector,) = await self.encode_documents([text])
        return vector

    def _embed(self, texts: list[str]) -> list[models.SparseVector]:
        """Посчитать векторы синхронно; вызывается из рабочего потока."""
        return [
            models.SparseVector(
                indices=[int(index) for index in embedding.indices],
                values=[float(value) for value in embedding.values],
            )
            for embedding in self._load().embed(texts, batch_size=self._batch_size)
        ]

    def _load(self) -> SparseTextEmbedding:
        """Загрузить модель один раз, даже если её просят из разных потоков."""
        with self._load_lock:
            if self._model is None:
                self._model = SparseTextEmbedding(
                    model_name=self._model_name,
                    language=self._language,
                )
            return self._model
