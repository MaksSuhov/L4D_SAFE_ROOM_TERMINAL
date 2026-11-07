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


class VectorStoreError(ApplicationError):
    """Поднимается, когда векторное хранилище недоступно или отвечает не тем,
    что объявляет его схема.
    """


class EvaluationError(ApplicationError):
    """Поднимается, когда стенд оценки не может прочитать кейс или оценить ответ."""


class AuthenticationError(ApplicationError):
    """Поднимается, когда запрос не подтвердил право на доступ.

    Каждое такое исключение несёт машинный код: клиент по нему отличает «истёк
    срок токена, обнови его» от «предъявлено не то, иди на экран входа», а текст
    сообщения при этом остаётся коротким и не подсказывает, существовала ли
    учётная запись.
    """

    code: str = "unauthenticated"


class InvalidCredentialsError(AuthenticationError):
    """Поднимается, когда пары «почта — пароль» не хватило для входа."""

    code = "invalid_credentials"

    def __init__(self) -> None:
        super().__init__("Неверная почта или пароль.")


class AccountInactiveError(AuthenticationError):
    """Поднимается, когда учётная запись существует, но заблокирована."""

    code = "account_inactive"

    def __init__(self) -> None:
        super().__init__("Учётная запись заблокирована.")


class InvalidTokenError(AuthenticationError):
    """Поднимается, когда токен не разбирается или подписан не нами."""

    code = "invalid_token"

    def __init__(self, reason: str = "Токен недействителен.") -> None:
        super().__init__(reason)


class TokenExpiredError(AuthenticationError):
    """Поднимается, когда подпись верна, но срок жизни токена истёк."""

    code = "expired_token"

    def __init__(self, reason: str = "Срок действия токена истёк.") -> None:
        super().__init__(reason)


class RefreshTokenReuseError(AuthenticationError):
    """Поднимается, когда уже погашенный refresh-токен предъявили снова.

    Это признак кражи: либо токен увели у законного клиента, либо законный клиент
    предъявил копию, снятую до ротации. Разбираться, кто из них кто, поздно,
    поэтому ответ один — вся цепочка ротаций гасится.
    """

    code = "refresh_token_reused"

    def __init__(self) -> None:
        super().__init__("Refresh-токен уже был использован; сессия отозвана.")


class AuthorizationError(ApplicationError):
    """Поднимается, когда личность установлена, но прав на действие нет."""

    code = "forbidden"

    def __init__(self, reason: str = "Недостаточно прав для этого действия.") -> None:
        super().__init__(reason)


class EmailAlreadyRegisteredError(ApplicationError):
    """Поднимается, когда на эту почту уже заведена учётная запись."""

    code = "email_taken"

    def __init__(self, email: str) -> None:
        super().__init__(f"Учётная запись с адресом {email} уже существует.")


class WeakPasswordError(ApplicationError):
    """Поднимается, когда пароль не отвечает политике учётных записей."""

    code = "weak_password"
