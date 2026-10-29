"""Strongly typed application configuration.

Every tunable value of the service lives here and is loaded from environment
variables (or a local ``.env`` file). The rest of the code depends on the
:class:`Settings` object instead of reading ``os.environ`` directly, which keeps
the configuration layer the single source of truth.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

from pydantic import Field, field_validator
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

    # --- OpenAI ----------------------------------------------------------
    openai_api_key: str = Field(default="", repr=False)
    openai_base_url: str | None = Field(
        default=None,
        description="Optional custom endpoint, useful for proxies and local gateways.",
    )
    openai_timeout_seconds: float = 30.0
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    chat_model: str = "gpt-4o-mini"
    chat_temperature: float = 0.7
    chat_max_tokens: int = 600

    # --- Retrieval augmented generation ----------------------------------
    rag_top_k: int = Field(default=3, ge=1, le=20)
    rag_min_similarity: float = Field(
        default=0.15,
        ge=-1.0,
        le=1.0,
        description="Cosine similarity floor; chunks below it are not passed to the model.",
    )

    @field_validator("api_prefix")
    @classmethod
    def _normalise_api_prefix(cls, value: str) -> str:
        """Accept ``api/v1``, ``/api/v1`` and ``/api/v1/`` and store ``/api/v1``."""
        stripped = value.strip().strip("/")
        return f"/{stripped}" if stripped else ""

    @property
    def is_openai_configured(self) -> bool:
        """Whether an API key is present, checked before the client is built."""
        return bool(self.openai_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    The value is cached because ``Settings`` parses the environment on every
    instantiation; tests clear the cache with ``get_settings.cache_clear()``.
    """
    return Settings()
