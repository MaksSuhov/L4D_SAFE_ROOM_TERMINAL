"""Схемы, описывающие контракт архива лора.

Через конвейер проходят три формы:

* :class:`LoreChunkCreate` — нормализованная запись корпуса, до индексации.
  Это ещё и формат строки снимка корпуса на диске.
* :class:`EmbeddedLoreChunk` — та же запись с её вектором и хешем содержимого,
  именно её и сохраняет репозиторий.
* :class:`LoreSource` — доказательства, возвращаемые клиенту, помеченные
  косинусной близостью, по которой они найдены.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.taxonomy import LoreTopic, SourceType

CONTENT_HASH_LENGTH = 64
"""Длина шестнадцатеричного дайджеста SHA-256, совпадающая со столбцом ``content_hash``."""


class LoreChunkCreate(BaseModel):
    """Фрагмент лора, ждущий индексации и сохранения."""

    content: str = Field(min_length=1)
    source_type: SourceType = Field(
        description=(
            "Уровень доказательств: подтверждённый лор, настенные надписи или "
            "реплика персонажа."
        ),
    )
    topic: LoreTopic = Field(description="Раздел вики, из которого извлечена запись.")
    title: str = Field(
        min_length=1,
        max_length=256,
        description="Источник в человекочитаемом виде, например «No Mercy › The Subway».",
    )
    character_name: str | None = Field(
        default=None,
        description="Кому принадлежит реплика; заполняется только для образцов голоса.",
    )
    source_url: str = Field(default="", max_length=512)


class EmbeddedLoreChunk(LoreChunkCreate):
    """Запись корпуса вместе с её эмбеддингом и хешем дедупликации."""

    content_hash: str = Field(min_length=CONTENT_HASH_LENGTH, max_length=CONTENT_HASH_LENGTH)
    embedding: list[float] = Field(min_length=1)


class LoreSource(BaseModel):
    """Найденный фрагмент, возвращаемый рядом со сгенерированным ответом."""

    id: int
    source_type: SourceType
    topic: LoreTopic
    title: str
    character_name: str | None = None
    content: str
    similarity: float = Field(ge=-1.0, le=1.0)


class IngestionReport(BaseModel):
    """Итог одного прогона индексации, показываемый CLI и тестами."""

    stored: int = 0
    skipped: int = 0
    by_source_type: dict[str, int] = Field(default_factory=dict)

    @property
    def total(self) -> int:
        """Число записей, предложенных прогону: сохранённых или пропущенных."""
        return self.stored + self.skipped
