"""Строго типизированная конфигурация приложения.

Здесь живут все настраиваемые параметры сервиса; они загружаются из переменных
окружения (или из локального файла ``.env``). Остальной код зависит от объекта
:class:`Settings`, а не читает ``os.environ`` напрямую, — так слой конфигурации
остаётся единственным источником правды.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Final

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

DEFAULT_EMBEDDING_DIMENSIONS: Final[int] = 1536
"""Размерность векторов ``text-embedding-3-small``; используется и столбцом pgvector."""


class Settings(BaseSettings):
    """Конфигурация Safe-Room Terminal во время выполнения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # --- Observability (Langfuse) -----------------------------------------
    # Tracing is opt-in through the credentials: without a key pair the terminal
    # builds the no-op tracer and nothing leaves the process. The field names
    # match the environment names, so LANGFUSE_* in .env needs no extra wiring.
    langfuse_public_key: str = Field(
        default="",
        repr=False,
        description="Публичный ключ проекта Langfuse; пара ключей включает трейсинг.",
    )
    langfuse_secret_key: str = Field(
        default="",
        repr=False,
        description="Секретный ключ проекта Langfuse.",
    )
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        description="Адрес Langfuse API; своя инсталляция указывает свой.",
    )
    langfuse_enabled: bool = Field(
        default=True,
        description="Выключатель наблюдаемости: false — работать с no-op трейсером.",
    )

    # --- Application -----------------------------------------------------
    app_name: str = "Left 4 Dead Interactive Safe-Room Terminal"
    app_version: str = "0.1.0"
    debug: bool = False
    api_prefix: str = Field(
        default="/api/v1",
        description="Префикс всех маршрутов; пустая строка — отдавать маршруты от корня.",
    )
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"],
        description=(
            "Origin браузеров, которым разрешено обращаться к API. По умолчанию покрыт "
            "dev-сервер Vite из комплектного фронтенда; перебивается списком через запятую."
        ),
    )

    # --- Database --------------------------------------------------------
    database_url: str = Field(
        default="postgresql+asyncpg://l4d:l4d@localhost:5432/l4d_terminal",
        description="Асинхронный DSN SQLAlchemy. В целевой базе нужно расширение pgvector.",
    )
    database_echo: bool = False
    database_pool_size: int = 5
    database_max_overflow: int = 10
    auto_create_tables: bool = Field(
        default=False,
        description="Создавать схему при старте приложения вместо запуска seed.py.",
    )

    # --- AI provider ------------------------------------------------------
    # Every provider is reached through the OpenAI-compatible API, so one set of
    # settings serves OpenAI, an aggregator such as RouterAI, or a local gateway.
    # Each field accepts several environment names; the first one that is present
    # wins, which keeps the shared names ahead of the provider-specific ones.
    ai_api_key: str = Field(
        default="",
        repr=False,
        validation_alias=AliasChoices("AI_API_KEY", "OPENAI_API_KEY", "ROUTERAI_API_KEY"),
    )
    ai_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AI_BASE_URL", "OPENAI_BASE_URL", "ROUTERAI_BASE_URL"),
        description="Адрес OpenAI-совместимого шлюза; если не задан, запрос идёт в сам OpenAI.",
    )
    ai_timeout_seconds: float = Field(
        default=30.0,
        validation_alias=AliasChoices("AI_TIMEOUT_SECONDS", "OPENAI_TIMEOUT_SECONDS"),
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description=(
            "Идентификатор модели у конкретного провайдера. Агрегаторы вроде RouterAI "
            "требуют префикс вендора, например openai/text-embedding-3-small."
        ),
    )
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    chat_model: str = "gpt-4o-mini"
    chat_temperature: float = 0.7
    chat_max_tokens: int = 600

    # --- Answer language --------------------------------------------------
    # The terminal answers in Russian regardless of the persona and of the
    # language of the question: the instruction below is appended to every
    # system prompt, so the language is a deployment-wide setting rather than a
    # property of each character stored in the database.
    answer_language_instruction: str = Field(
        default=(
            "Всегда отвечай только на русском языке, независимо от языка вопроса. "
            "Весь диалог, включая каждую реплику, веди на русском."
        ),
        description=(
            "Инструкция, дописываемая к промпту каждого образа, чтобы закрепить язык "
            "ответов и всего диалога."
        ),
    )

    # --- Wiki corpus ------------------------------------------------------
    # The corpus is downloaded once from the Left 4 Dead Fandom wiki, parsed and
    # written to ``corpus_path`` as JSONL. Seeding reads that snapshot, so it is
    # reproducible and works offline; only ``--refresh`` talks to the network.
    wiki_base_url: str = "https://left4dead.fandom.com"
    wiki_request_timeout_seconds: float = Field(default=30.0, gt=0.0)
    wiki_user_agent: str = Field(
        default="L4D-SafeRoom-Terminal/0.1 (portfolio project; lore ingestion)",
        description="Отправляется с каждым запросом к вики: API Fandom не пускает анонимов.",
    )
    wiki_cache_dir: Path = Field(
        default=Path("data/wiki_cache"),
        description="Сырой викитекст, сохраняемый между запусками: повторный разбор без сети.",
    )
    wiki_cache_enabled: bool = True
    corpus_path: Path = Field(
        default=Path("data/corpus/lore.jsonl"),
        description="Снимок нормализованного корпуса, загружаемый в векторное хранилище.",
    )
    ingestion_batch_size: int = Field(
        default=64,
        ge=1,
        le=2048,
        description="Сколько текстов уходит в один запрос на эмбеддинги при индексации.",
    )
    translation_batch_size: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Сколько текстов уходит в один запрос на перевод при нормализации языка.",
    )
    corpus_language: str = Field(
        default="ru",
        min_length=2,
        max_length=8,
        description=(
            "Язык, в который нормализуется корпус при индексации. Совпадение с языком "
            "ответов делает поиск одноязычным, а это измеримо поднимает близость по "
            "сравнению с кросс-язычным поиском."
        ),
    )

    # --- Retrieval augmented generation ----------------------------------
    # Retrieval is tiered: verified lore and wall graffiti are looked up
    # separately, each with its own budget, so a rumour can never crowd out a
    # fact (or the other way round).
    rag_official_top_k: int = Field(default=4, ge=0, le=20)
    rag_graffiti_top_k: int = Field(default=3, ge=0, le=20)
    rag_voice_top_k: int = Field(default=2, ge=0, le=10)
    rag_min_similarity: float = Field(
        default=0.25,
        ge=-1.0,
        le=1.0,
        description=(
            "Порог косинусной близости для уровней-доказательств (лор и надписи): "
            "более слабые совпадения отбрасываются и не попадают в модель. Чуть выше "
            "уровня шума text-embedding-3-small, где несвязанные тексты редко "
            "проходят 0.2."
        ),
    )
    rag_voice_min_similarity: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Порог для собственных реплик персонажа. Они служат образцом стиля, а не "
            "доказательством, поэтому самые близкие передаются даже когда вопрос не "
            "связан с тем, для чего эти реплики записаны."
        ),
    )

    # --- HIDE: hypothetical expansion of the question ---------------------
    # A question may share no wording with the record that answers it: the
    # player asks about a character, while the record lives in the article of
    # the campaign that character appears in. HIDE asks the model for
    # hypothetical formulations (and one hypothetical article fragment) that
    # bridge those entities, searches each of them, and fuses the matches by
    # similarity — the budgets and floors of the tiers keep their meaning.
    hide_enabled: bool = Field(
        default=True,
        description=(
            "Искать по гипотетическим формулировкам (HIDE), а не только по исходному "
            "вопросу. Отключается, когда важнее время до первого токена: расширение "
            "добавляет один запрос к провайдеру перед поиском."
        ),
    )
    hide_max_questions: int = Field(
        default=4,
        ge=0,
        le=10,
        description=(
            "Сколько формулировок HIDE вправе добавить к исходному вопросу; 0 — "
            "не добавлять ни одной."
        ),
    )

    # --- Offline evaluation ----------------------------------------------
    # The scoring stand reports numbers that only mean something if two runs of
    # the same dataset are comparable. Generation therefore runs at a pinned
    # temperature instead of the chat default: sampling at 0.7 made the same
    # dataset score differently on every run.
    eval_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description=(
            "Температура генерации на стенде оценки. Закреплена низкой, чтобы один и "
            "тот же датасет давал сравнимые числа между прогонами."
        ),
    )
    eval_attempts: int = Field(
        default=2,
        ge=1,
        le=5,
        description=(
            "Сколько раз стенд повторяет кейс, если модель вернула пустой ответ или "
            "запрос упал. Кейс, не давший ответ за все попытки, попадает в отчёт как сбой."
        ),
    )

    @field_validator("api_prefix")
    @classmethod
    def _normalise_api_prefix(cls, value: str) -> str:
        """Привести ``api/v1``, ``/api/v1`` и ``/api/v1/`` к виду ``/api/v1``."""
        stripped = value.strip().strip("/")
        return f"/{stripped}" if stripped else ""

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Принять ``a,b``, ``a, b`` или JSON-список и привести к списку."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def is_ai_configured(self) -> bool:
        """Есть ли API-ключ; проверяется до создания клиента."""
        return bool(self.ai_api_key.strip())

    @property
    def is_langfuse_configured(self) -> bool:
        """Есть ли пара ключей Langfuse; проверяется до создания трейсера."""
        return bool(self.langfuse_public_key.strip() and self.langfuse_secret_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Вернуть экземпляр настроек, общий для всего процесса.

    Значение кэшируется, потому что ``Settings`` разбирает окружение при каждом
    создании; тесты сбрасывают кэш через ``get_settings.cache_clear()``.
    """
    return Settings()
