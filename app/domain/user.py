from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.taxonomy import UserRole


def normalise_email(email: str) -> str:
    """Привести адрес почты к виду, в котором он служит логином.

    Регистр в адресе значим только теоретически, а на практике «Ellis@…» и
    «ellis@…» — один ящик, и заводить на них две учётные записи нельзя.
    """
    return email.strip().lower()


@dataclass(slots=True)
class User:
    """Учётная запись терминала.

    Хранится ровно то, что нужно, чтобы человека узнать и решить, что ему
    разрешено: адрес почты как логин, хеш пароля и роль. Самого пароля нет ни
    здесь, ни в хранилище, поэтому утечка записи не выдаёт учётные данные.
    """

    email: str
    """Адрес почты, приведённый к нижнему регистру: он же логин и ключ уникальности."""
    password_hash: str
    """Готовый к проверке хеш пароля; алгоритм и его параметры зашиты в саму строку."""
    display_name: str
    role: UserRole = UserRole.MEMBER
    is_active: bool = True
    id: str | None = None
    """Идентификатор записи; до сохранения его нет, его выдаёт хранилище."""
    created_at: datetime | None = None

    @property
    def identifier(self) -> str:
        """Идентификатор сохранённой записи.

        Обращаться к нему до сохранения не за чем, поэтому несохранённая запись
        обязана падать здесь, а не уезжать дальше с пустым значением.
        """
        if self.id is None:
            raise ValueError("the user has not been stored yet and has no identifier")
        return self.id

    @property
    def is_admin(self) -> bool:
        """Правда ли, что учётной записи разрешены административные действия."""
        return self.role is UserRole.ADMIN
