"""Schemas describing the chat request/response contract."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.schemas.lore_schema import LoreSource


class ChatRequest(BaseModel):
    """Incoming question addressed to a character."""

    character_id: int = Field(gt=0, description="Identifier of the answering character.")
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def _reject_blank_message(cls, value: str) -> str:
        normalised = value.strip()
        if not normalised:
            raise ValueError("message must not be blank")
        return normalised


class ChatResponse(BaseModel):
    """Answer produced by the RAG pipeline together with its evidence."""

    character_id: int
    character_name: str
    answer: str
    model: str
    sources: list[LoreSource] = Field(default_factory=list)
