from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.domain import MAX_PASSWORD_LENGTH, MIN_PASSWORD_LENGTH
from app.schemas.user_schema import UserRead


class RegisterRequest(BaseModel):
    """Заявка на учётную запись на терминале.

    Границы длины пароля берутся из доменной политики, а не задаются здесь
    заново: схема обязана отвергать то же, что и сервис, иначе одно из двух
    правил однажды разойдётся с другим.
    """

    email: EmailStr = Field(description="Адрес почты; он же логин и ключ уникальности.")
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)
    display_name: str | None = Field(
        default=None,
        max_length=64,
        description="Подпись на терминале; пустая — соберётся из адреса почты.",
    )

    @field_validator("display_name")
    @classmethod
    def _blank_name_is_absent(cls, value: str | None) -> str | None:
        """Считать подпись из пробелов отсутствующей."""
        if value is None:
            return None
        return value.strip() or None


class RefreshRequest(BaseModel):
    """Refresh-токен, переданный в теле.

    Нужен клиентам без cookie: браузер прикладывает свой токен сам, а серверный
    вызов обязан назвать его явно. Тело необязательно, поэтому ``None`` здесь
    означает «взять токен из cookie».
    """

    refresh_token: str | None = None


class SessionResponse(BaseModel):
    """Итог установления сессии.

    Самих токенов в теле нет: они уезжают в httpOnly-cookie. Клиент получает то,
    что ему и нужно знать, — кто вошёл, сколько живёт токен доступа и CSRF-токен,
    который он обязан повторять заголовком при изменяющих запросах.
    """

    user: UserRead
    csrf_token: str
    expires_in: int = Field(gt=0, description="Срок жизни токена доступа в секундах.")
    token_type: str = "Bearer"
