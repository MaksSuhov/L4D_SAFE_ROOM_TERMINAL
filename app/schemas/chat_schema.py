"""Схемы, описывающие контракт запроса и ответа чата."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.schemas.lore_schema import LoreSource


class ChatRequest(BaseModel):
    """Входящий вопрос, адресованный персонажу."""

    character_id: int = Field(gt=0, description="Идентификатор персонажа, который отвечает.")
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def _reject_blank_message(cls, value: str) -> str:
        normalised = value.strip()
        if not normalised:
            raise ValueError("message must not be blank")
        return normalised


class ChatResponse(BaseModel):
    """Ответ, сгенерированный RAG-конвейером, вместе с его доказательствами."""

    character_id: int
    character_name: str
    answer: str
    model: str
    sources: list[LoreSource] = Field(default_factory=list)
