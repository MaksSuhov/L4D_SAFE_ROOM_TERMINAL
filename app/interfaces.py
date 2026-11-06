"""Порты приложения (принцип инверсии зависимостей).

Слой сервисов зависит от этих определений ``Protocol``, а не от конкретных
адаптеров Qdrant или обёртки над OpenAI SDK. Вместо них можно внедрить
любой объект подходящей формы — настоящий адаптер, заглушку из тестов или
будущий клиент иного векторного хранилища.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

from app.domain import Character, LoreChunk
from app.schemas import EmbeddedLoreChunk, LoreSource
from app.taxonomy import SourceType


class RetrievedChunk(NamedTuple):
    """Фрагмент лора вместе с косинусной близостью поискового запроса."""

    chunk: LoreChunk
    similarity: float


@dataclass(frozen=True, slots=True)
class PreparedChat:
    """Всё, что нужно одному ходу чата после выполнения выборки.

    Это стык между двумя половинами конвейера: выборка его заполняет
    (вопрос -> доказательства -> брифинг), а генерации нужно только то, что здесь
    хранится. Открыв его наружу, вызывающий код может стримить ответ и сохранять
    доказательства — именно так и поступает офлайн-стенд оценки.
    """

    character_id: int
    character_name: str
    model: str
    system_prompt: str
    context: str
    few_shot_examples: Sequence[Mapping[str, str]]
    sources: Sequence[LoreSource]


@dataclass(frozen=True, slots=True)
class ExpandedQuery:
    """Вопрос игрока как набор поисковых формулировок.

    Исходный вопрос всегда идёт первым: расширение добавляет к нему формулировки,
    но никогда не подменяет им вопрос, поэтому выключенный или неудавшийся
    расширитель оставляет поиск ровно таким, каким он был без него.
    """

    original: str
    hypotheticals: tuple[str, ...] = ()

    @property
    def probes(self) -> tuple[str, ...]:
        """Формулировки, по которым пойдёт поиск: исходный вопрос первым."""
        return (self.original, *self.hypotheticals)

    @property
    def is_expanded(self) -> bool:
        """Правда ли, что к вопросу добавились гипотетические формулировки."""
        return bool(self.hypotheticals)


class CharacterRepositoryProtocol(Protocol):
    """Доступ к образам персонажей и их сохранение."""

    async def get_by_id(self, entity_id: int) -> Character | None:
        """Вернуть персонажа с данным идентификатором или ``None``."""
        ...

    async def get_by_name(self, name: str) -> Character | None:
        """Найти персонажа по его уникальному отображаемому имени."""
        ...

    async def get_all(self) -> Sequence[Character]:
        """Вернуть всех сохранённых персонажей, упорядоченных по идентификатору."""
        ...

    async def save(self, character: Character) -> bool:
        """Сохранить образ, обновив его на месте, если он уже есть.

        Возвращает признак того, что образ добавлен впервые: инструменту
        наполнения это нужно для отчёта о прогоне. Обновление на месте — часть
        контракта, а не деталь реализации: правка промпта не должна требовать
        очистки всего хранилища.
        """
        ...


class LoreChunkRepositoryProtocol(Protocol):
    """Сохранение и векторный поиск по фрагментам лора."""

    async def create_chunk(self, record: EmbeddedLoreChunk) -> LoreChunk:
        """Сохранить один проиндексированный фрагмент лора."""
        ...

    async def create_chunks(self, records: Sequence[EmbeddedLoreChunk]) -> Sequence[LoreChunk]:
        """Сохранить несколько проиндексированных фрагментов лора за один запрос."""
        ...

    async def search_similar_chunks(
        self,
        query_embedding: Sequence[float],
        query_text: str,
        *,
        limit: int,
        source_types: Sequence[SourceType] | None = None,
        topic: str | None = None,
        character_name: str | None = None,
    ) -> Sequence[RetrievedChunk]:
        """Вернуть ``limit`` ближайших фрагментов по убыванию близости.

        ``source_types`` ограничивает поиск одним или несколькими уровнями
        доказательств — именно это делает двухуровневую выборку возможной: факты
        ищутся среди официальных записей, а слухи — среди надписей на стенах.

        ``query_text`` идёт рядом с вектором, потому что гибридный поиск ведут два
        плеча: косинусное и лексическое. Вернуть текст из эмбеддинга нельзя,
        поэтому вопрос передаётся дословно, как его набрал игрок. Реализация без
        лексического плеча вправе его проигнорировать.

        Реализация сама решает, кто попадёт в бюджет уровня: гибридная наполняет
        его в порядке слияния двух плеч, а ``similarity`` остаётся косинусной
        близостью, чтобы вызывающий мог фильтровать по ней.
        """
        ...

    async def existing_content_hashes(self, hashes: Sequence[str]) -> set[str]:
        """Вернуть, какие из данных хешей содержимого уже сохранены."""
        ...

    async def count(self) -> int:
        """Вернуть число сохранённых фрагментов лора."""
        ...


class LoreArchiveAdminProtocol(LoreChunkRepositoryProtocol, Protocol):
    """Операции жизненного цикла архива, доступные только инструменту наполнения.

    Отдельный порт существует ради одной строки в типах: сервис, зависящий от
    :class:`LoreChunkRepositoryProtocol`, физически не может опустошить архив,
    потому что такого метода в его контракте нет. Право на очистку есть только у
    инструмента, владеющего жизненным циклом архива, — у seed-скрипта.
    """

    async def delete_all(self) -> int:
        """Удалить все записи архива и сообщить, сколько их было."""
        ...


class AIClientProtocol(Protocol):
    """Доступ к провайдеру языковой модели."""

    model_name: str
    """Имя чат-модели; показывается в ответах API."""

    async def get_embedding(self, text: str) -> list[float]:
        """Посчитать эмбеддинг ``text`` в вектор из ``embedding_dimensions`` чисел."""
        ...

    async def get_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """Посчитать эмбеддинги пакета текстов, сохранив их порядок.

        Пакетность важна: корпус содержит сотни записей, и запрос на каждую запись
        превратил бы наполнение базы в перерыв на кофе.
        """
        ...

    async def translate_texts(self, texts: Sequence[str]) -> list[str]:
        """Перевести пакет текстов на заданный язык корпуса.

        Используется один раз, пока строится корпус, так что английская вики
        превращается в корпус на том языке, на котором отвечает терминал. Реализации
        обязаны вернуть ровно один перевод на вход, откатываясь к исходному тексту
        вместо того, чтобы выбросить запись.
        """
        ...

    async def generate_response(
        self,
        system_prompt: str,
        context: str,
        user_message: str,
        few_shot_examples: Sequence[Mapping[str, str]] | None = None,
    ) -> str:
        """Сгенерировать ответ ассистента, опираясь на ``context``."""
        ...

    def stream_response(
        self,
        system_prompt: str,
        context: str,
        user_message: str,
        few_shot_examples: Sequence[Mapping[str, str]] | None = None,
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Отдавать ответ по фрагментам, в том порядке, в котором их шлёт провайдер.

        Тот же ответ, что и у :meth:`generate_response`, только доставляемый потоком
        фрагментов. Стриминг делает наблюдаемым время до первого токена: пока не
        пришёл первый фрагмент, ничего ещё не сгенерировано. ``temperature``
        переопределяет настройку на один вызов, ``None`` оставляет значение по
        умолчанию.
        """
        ...

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.0,
    ) -> str:
        """Ответить на одну инструкцию, без выборки и без персонажей.

        Низкоуровневая точка входа для задач, которым нужен только ответ модели:
        оценить ответ, классифицировать текст, извлечь поле. Температура здесь
        аргумент, потому что такие задачи хотят детерминизма, а не творчества.
        """
        ...


class QueryExpanderProtocol(Protocol):
    """Разворачивает вопрос игрока в набор поисковых формулировок.

    Нужен там, где вопрос и ответ на него не делят ни одного слова: игрок
    спрашивает про персонажа, а запись лежит в статье про кампанию, где этот
    персонаж упомянут. Расширитель обязан вернуть исходный вопрос даже тогда,
    когда сам не смог ничего добавить, — поиск от этого не должен ломаться.
    """

    async def expand(self, question: str) -> ExpandedQuery:
        """Развернуть ``question`` в формулировки, по которым пойдёт поиск."""
        ...


class TracingSpanProtocol(Protocol):
    """Один участок конвейера, как его видит система наблюдаемости.

    Сервис открывает участок и прикладывает к нему то, чего не видно из имени:
    что пришло на вход, что получилось на выходе, какие записи архива нашлись.
    Ничего из этого не обязательно — выключенный трейсер всё принимает и
    отбрасывает.
    """

    def annotate(
        self,
        *,
        input: Any = None,
        output: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Дописать к участку его вход, выход и подробности."""
        ...


class TracerProtocol(Protocol):
    """Размечает конвейер участками, которые потом видно в трейсе.

    Сервисы зависят от этого порта, а не от SDK наблюдаемости: так трейсинг
    остаётся деталью адаптера, выключается одной настройкой и не тянет внешний
    сервис в тесты. Владение клиентом тоже здесь — освобождает его тот, кто
    собрал граф зависимостей.
    """

    def span(
        self,
        name: str,
        *,
        as_type: str = "span",
    ) -> AbstractContextManager[TracingSpanProtocol]:
        """Открыть участок ``name``; закрывается на выходе из блока."""
        ...

    def flush(self) -> None:
        """Дождаться, пока накопленные участки уедут в хранилище."""
        ...

    def shutdown(self) -> None:
        """Освободить ресурсы трейсера."""
        ...
