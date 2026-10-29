"""Domain-level exceptions.

They carry no knowledge of HTTP; the API layer translates them into status
codes so that services stay framework agnostic.
"""

from __future__ import annotations


class ApplicationError(Exception):
    """Base class for every expected failure raised by the application."""


class ResourceNotFoundError(ApplicationError):
    """Raised when an entity referenced by a request does not exist."""

    def __init__(self, resource: str, identifier: int | str) -> None:
        self.resource = resource
        self.identifier = identifier
        super().__init__(f"{resource} with identifier {identifier!r} was not found")


class CharacterNotFoundError(ResourceNotFoundError):
    """Raised when the requested character persona is absent from the database."""

    def __init__(self, character_id: int) -> None:
        super().__init__("Character", character_id)


class AIServiceError(ApplicationError):
    """Raised when the OpenAI provider fails or is misconfigured."""
