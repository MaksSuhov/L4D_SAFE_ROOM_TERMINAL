"""Исключения уровня домена.

Они ничего не знают про HTTP; слой API переводит их в коды состояния, чтобы
сервисы оставались независимы от фреймворка.
"""

from __future__ import annotations


class ApplicationError(Exception):
    """Базовый класс для каждого ожидаемого сбоя, поднятого приложением."""


class ResourceNotFoundError(ApplicationError):
    """Поднимается, когда сущность, на которую ссылается запрос, не существует."""

    def __init__(self, resource: str, identifier: int | str) -> None:
        self.resource = resource
        self.identifier = identifier
        super().__init__(f"{resource} with identifier {identifier!r} was not found")


class CharacterNotFoundError(ResourceNotFoundError):
    """Поднимается, когда запрошенного образа персонажа нет в базе данных."""

    def __init__(self, character_id: int) -> None:
        super().__init__("Character", character_id)


class AIServiceError(ApplicationError):
    """Поднимается, когда провайдер OpenAI даёт сбой или настроен неверно."""


class EvaluationError(ApplicationError):
    """Поднимается, когда стенд оценки не может прочитать кейс или оценить ответ."""
