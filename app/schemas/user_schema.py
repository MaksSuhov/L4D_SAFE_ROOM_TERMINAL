from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator

from app.taxonomy import UserRole


class UserRead(BaseModel):
    """Представление учётной записи, возвращаемое API.

    Хеша пароля здесь нет и не будет: схема описывает то, что о человеке можно
    рассказать, а не то, чем подтверждается его личность.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    email: EmailStr
    display_name: str
    role: UserRole
    is_active: bool
    created_at: datetime


class UserUpdate(BaseModel):
    """Изменения учётной записи, которые вправе внести администратор."""

    role: UserRole | None = None
    is_active: bool | None = None

    @model_validator(mode="after")
    def _require_a_change(self) -> UserUpdate:
        """Отклонить запрос, который ничего не меняет."""
        if self.role is None and self.is_active is None:
            raise ValueError("укажите хотя бы одно поле: role или is_active")
        return self


class RevocationReport(BaseModel):
    """Итог отзыва сессий: сколько их было погашено."""

    revoked: int = Field(ge=0, description="Сколько сессий погашено этим запросом.")
