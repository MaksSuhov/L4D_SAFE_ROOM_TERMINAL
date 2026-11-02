"""Схемы, описывающие контракт образа персонажа."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FewShotExample(BaseModel):
    """Один обмен репликами user/assistant, внедрённый в историю чата."""

    user: str = Field(min_length=1)
    assistant: str = Field(min_length=1)


class CharacterBase(BaseModel):
    """Поля, общие для каждого представления персонажа."""

    name: str = Field(min_length=1, max_length=128)
    system_prompt: str = Field(min_length=1)
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)


class CharacterCreate(CharacterBase):
    """Payload, принимаемый инструментом наполнения базы."""


class CharacterRead(CharacterBase):
    """Представление персонажа, возвращаемое API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
