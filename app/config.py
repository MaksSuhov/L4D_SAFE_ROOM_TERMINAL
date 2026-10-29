"""Strongly typed application configuration.

Every tunable value of the service lives here and is loaded from environment
variables (or a local ``.env`` file). The rest of the code depends on the
:class:`Settings` object instead of reading ``os.environ`` directly, which keeps
the configuration layer the single source of truth.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_EMBEDDING_DIMENSIONS: Final[int] = 1536
"""Dimension of ``text-embedding-3-small`` vectors, also used by the pgvector column."""


class Settings(BaseSettings):
    """Runtime configuration of the Safe-Room Terminal."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # --- Application -----------------------------------------------------
    app_name: str = "Left 4 Dead Interactive Safe-Room Terminal"
    app_version: str = "0.1.0"
    debug: bool = False
    api_prefix: str = Field(
        default="/api/v1",
        description="Prefix for every route; set to an empty string to serve routes from the root.",
    )
    # --- Database --------------------------------------------------------
    database_url: str = Field(
        default="postgresql+asyncpg://l4d:l4d@localhost:5432/l4d_terminal",
        description="Async SQLAlchemy DSN. The target database must have the pgvector extension.",
    )
    database_echo: bool = False
    database_pool_size: int = 5
    database_max_overflow: int = 10
    auto_create_tables: bool = Field(
        default=False,
        description="Create the schema on application start-up instead of running seed.py.",
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
        description="Endpoint of the OpenAI-compatible gateway; unset targets OpenAI itself.",
    )
    ai_timeout_seconds: float = Field(
        default=30.0,
        validation_alias=AliasChoices("AI_TIMEOUT_SECONDS", "OPENAI_TIMEOUT_SECONDS"),
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description=(
            "Provider-specific model id. Aggregators such as RouterAI require the "
            "vendor prefix, e.g. openai/text-embedding-3-small."
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
            "Instruction appended to every persona prompt to pin the language of "
            "the answers and of the whole dialogue."
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
        description="Sent with every wiki request; the Fandom API rejects anonymous clients.",
    )
    wiki_cache_dir: Path = Field(
        default=Path("data/wiki_cache"),
        description="Raw wikitext kept between runs so a re-parse needs no network.",
    )
    wiki_cache_enabled: bool = True
    corpus_path: Path = Field(
        default=Path("data/corpus/lore.jsonl"),
        description="Normalised corpus snapshot ingested into the vector store.",
    )
    ingestion_batch_size: int = Field(
        default=64,
        ge=1,
        le=2048,
        description="Texts per embedding request during ingestion.",
    )
    translation_batch_size: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Texts per translation request when normalising the corpus language.",
    )
    corpus_language: str = Field(
        default="ru",
        min_length=2,
        max_length=8,
        description=(
            "Language the corpus is normalised into during ingestion. Keeping it in the "
            "same language as the answers makes retrieval monolingual, which measurably "
            "raises similarity scores compared with cross-language search."
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
            "Cosine similarity floor for the evidence tiers (lore and graffiti); "
            "weaker matches are dropped instead of being passed to the model. "
            "Slightly above the noise level of text-embedding-3-small, where "
            "unrelated texts rarely pass 0.2."
        ),
    )
    rag_voice_min_similarity: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Floor for the character's own lines. They are a style reference rather "
            "than evidence, so the most relevant ones are passed on even when the "
            "question is unrelated to what they were recorded for."
        ),
    )

    @field_validator("api_prefix")
    @classmethod
    def _normalise_api_prefix(cls, value: str) -> str:
        """Accept ``api/v1``, ``/api/v1`` and ``/api/v1/`` and store ``/api/v1``."""
        stripped = value.strip().strip("/")
        return f"/{stripped}" if stripped else ""

    @property
    def is_ai_configured(self) -> bool:
        """Whether an API key is present, checked before the client is built."""
        return bool(self.ai_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    The value is cached because ``Settings`` parses the environment on every
    instantiation; tests clear the cache with ``get_settings.cache_clear()``.
    """
    return Settings()
