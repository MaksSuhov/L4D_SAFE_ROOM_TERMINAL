"""Pydantic-схемы: контракт между слоем API и его клиентами.

ORM-модели никогда не покидают слой сервисов, они преобразуются в эти схемы,
что сохраняет контракт HTTP стабильным, даже если модель хранения изменится.
Фасад пакета реэкспортирует каждую схему, чтобы вызывающий код сохранял единый
путь импорта, пока определения сгруппированы по агрегату.
"""

from app.schemas.auth_schema import RefreshRequest, RegisterRequest, SessionResponse
from app.schemas.character_schema import (
    CharacterBase,
    CharacterCreate,
    CharacterRead,
    FewShotExample,
)
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.schemas.health_schema import HealthResponse
from app.schemas.lore_schema import (
    EmbeddedLoreChunk,
    IngestionReport,
    LoreChunkCreate,
    LoreSource,
)
from app.schemas.user_schema import RevocationReport, UserRead, UserUpdate

__all__ = [
    "CharacterBase",
    "CharacterCreate",
    "CharacterRead",
    "ChatRequest",
    "ChatResponse",
    "EmbeddedLoreChunk",
    "FewShotExample",
    "HealthResponse",
    "IngestionReport",
    "LoreChunkCreate",
    "LoreSource",
    "RefreshRequest",
    "RegisterRequest",
    "RevocationReport",
    "SessionResponse",
    "UserRead",
    "UserUpdate",
]
