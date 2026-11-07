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
from datetime import datetime
from typing import Any, NamedTuple, Protocol

from app.domain import Character, LoreChunk, RefreshSession, User
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


@dataclass(frozen=True, slots=True)
class AccessTokenClaims:
    """Разобранное содержимое токена доступа.

    Вместе с личностью в токен кладутся адрес и роль: их достаточно, чтобы
    отличить администратора от выжившего, не поднимая учётную запись из
    хранилища, — а запись всё равно читается, если токен нужно проверить целиком.
    """

    subject: str
    email: str
    role: str
    issued_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class IssuedAccessToken:
    """Подписанный токен доступа вместе с моментом, когда он перестанет работать."""

    value: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class MintedRefreshToken:
    """Новый refresh-токен: сырое значение уходит клиенту, отпечаток — в хранилище."""

    value: str
    digest: str


class ClockProtocol(Protocol):
    """Источник текущего времени.

    Сервис учётных записей зависит от этих часов, а не от ``datetime.now``:
    сроки жизни токенов проверяются во многих тестах, и подменяемые часы
    позволяют их не ждать.
    """

    def now(self) -> datetime:
        """Вернуть текущий момент в UTC с указанием часового пояса."""
        ...


class PasswordHasherProtocol(Protocol):
    """Превращение пароля в проверяемый хеш и обратно — в ответ «подходит ли»."""

    def hash(self, password: str) -> str:
        """Посчитать хеш пароля вместе с солью и параметрами алгоритма."""
        ...

    def verify(self, password: str, encoded: str) -> bool:
        """Проверить пароль против ранее сохранённого хеша."""
        ...

    def needs_rehash(self, encoded: str) -> bool:
        """Правда ли, что хеш посчитан с устаревшими параметрами и его пора пересчитать."""
        ...

    def equalise_timing(self, password: str) -> None:
        """Проделать работу проверки, ничего не проверяя.

        Нужно входу: когда почты в хранилище нет, проверять нечего, и ответ
        вернулся бы заметно быстрее, чем при неверном пароле, — по этой разнице
        учётные записи и перечисляют. Вызов стоит столько же, сколько настоящая
        проверка, и ничего о ней не сообщает.
        """
        ...


class AccessTokenIssuerProtocol(Protocol):
    """Выдача и разбор токенов доступа."""

    def issue(self, user: User) -> IssuedAccessToken:
        """Подписать токен доступа, удостоверяющий эту учётную запись."""
        ...

    def read(self, token: str) -> AccessTokenClaims:
        """Разобрать токен и вернуть его содержимое.

        Поднимает :class:`~app.exceptions.TokenExpiredError`, если подпись верна,
        но срок вышел, и :class:`~app.exceptions.InvalidTokenError` — если токен
        подделан, испорчен или выпущен не нами.
        """
        ...


class RefreshTokenFactoryProtocol(Protocol):
    """Выдача refresh-токенов и их отпечатков."""

    def mint(self) -> MintedRefreshToken:
        """Выдать новый токен вместе с отпечатком, пригодным для хранения."""
        ...

    def digest(self, token: str) -> str:
        """Посчитать отпечаток предъявленного токена, чтобы найти его сессию."""
        ...


class UserRepositoryProtocol(Protocol):
    """Хранение и чтение учётных записей."""

    async def get_by_id(self, entity_id: str) -> User | None:
        """Вернуть учётную запись с данным идентификатором или ``None``."""
        ...

    async def get_by_email(self, email: str) -> User | None:
        """Найти учётную запись по адресу почты, приведённому к нижнему регистру."""
        ...

    async def get_all(self) -> Sequence[User]:
        """Вернуть все учётные записи, упорядоченные по адресу почты."""
        ...

    async def create(self, user: User) -> None:
        """Сохранить новую запись, выдав ей идентификатор и время создания.

        Уникальность адреса проверяет вызывающий: хранилище не умеет удерживать
        его само, и об этом честнее сказать здесь, чем делать вид, что умеет.
        """
        ...

    async def update(self, user: User) -> None:
        """Записать изменённые поля существующей учётной записи."""
        ...

    async def delete(self, entity_id: str) -> bool:
        """Удалить учётную запись и сообщить, была ли она вообще."""
        ...

    async def count(self) -> int:
        """Вернуть число учётных записей."""
        ...


class RefreshSessionRepositoryProtocol(Protocol):
    """Хранение и отзыв сессий, выданных по refresh-токенам.

    Время сюда приходит аргументом, а не берётся внутри: часы принадлежат
    сервису, и один и тот же момент должен стоять и в записи, и в её отзыве.
    """

    async def create(self, session: RefreshSession) -> None:
        """Сохранить сессию, выдав ей идентификатор."""
        ...

    async def get_by_digest(self, digest: str) -> RefreshSession | None:
        """Найти сессию по отпечатку предъявленного токена."""
        ...

    async def revoke(
        self,
        session_id: str,
        *,
        revoked_at: datetime,
        replaced_by: str | None = None,
    ) -> None:
        """Погасить одну сессию, оставив след о том, кто её заменил."""
        ...

    async def revoke_family(self, family_id: str, *, revoked_at: datetime) -> int:
        """Погасить всю цепочку ротаций и сообщить, сколько сессий было отозвано."""
        ...

    async def revoke_all_for_user(self, user_id: str, *, revoked_at: datetime) -> int:
        """Погасить все сессии учётной записи и сообщить, сколько их было."""
        ...
