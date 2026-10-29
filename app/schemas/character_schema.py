"""Schemas describing the character persona contract."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FewShotExample(BaseModel):
    """A single user/assistant exchange injected into the chat history."""

    user: str = Field(min_length=1)
    assistant: str = Field(min_length=1)


class CharacterBase(BaseModel):
    """Fields shared by every representation of a character."""

    name: str = Field(min_length=1, max_length=128)
    system_prompt: str = Field(min_length=1)
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)


class CharacterCreate(CharacterBase):
    """Payload accepted by the seeding tooling."""


class CharacterRead(CharacterBase):
    """Character representation returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
