"""Заведение учётной записи терминала из командной строки.

Первый оператор не может появиться через API: самостоятельная регистрация всегда
заводит выжившего, а роль назначает тот, у кого это обязанность. Поэтому первая
учётная запись заводится отсюда::

    make user EMAIL=operator@l4d.dev
    .venv/bin/python -m scripts.create_user --email operator@l4d.dev --role admin

Пароль спрашивается дважды и не отображается; ``--password`` оставлен для
неинтерактивных запусков. Повторный запуск на занятый адрес ничего не делает, а
``--force`` приводит существующую запись к заданным роли и паролю.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys

from app.config import Settings, get_settings
from app.dependencies import (
    build_access_token_issuer,
    build_auth_service,
    build_password_hasher,
    build_refresh_session_repository,
    build_refresh_token_factory,
    build_user_repository,
)
from app.domain import User, normalise_email
from app.exceptions import ApplicationError, EmailAlreadyRegisteredError
from app.interfaces import PasswordHasherProtocol, UserRepositoryProtocol
from app.services import SystemClock
from app.taxonomy import UserRole
from app.vector import VectorStore

ROLES = {role.value: role for role in UserRole}


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Завести учётную запись терминала.")
    parser.add_argument("--email", required=True, help="адрес почты; он же логин")
    parser.add_argument(
        "--role",
        choices=sorted(ROLES),
        default=UserRole.ADMIN.value,
        help="роль учётной записи (по умолчанию оператор)",
    )
    parser.add_argument("--name", default=None, help="подпись на терминале")
    parser.add_argument(
        "--password",
        default=None,
        help="пароль; без него он будет запрошен и не отобразится",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="привести существующую запись к заданным роли и паролю",
    )
    return parser.parse_args()


def request_password(explicit: str | None) -> str:
    """Получить пароль: из аргумента или двойным запросом без эха."""
    if explicit is not None:
        return explicit

    if not sys.stdin.isatty():
        raise ApplicationError(
            "Пароль не запросить без терминала: передайте его аргументом --password."
        )

    password = getpass.getpass("Пароль: ")
    confirmation = getpass.getpass("Повторите пароль: ")
    if password != confirmation:
        raise ApplicationError("Пароли не совпали.")
    return password


async def create_account(
    settings: Settings,
    *,
    email: str,
    password: str,
    role: UserRole,
    display_name: str | None,
    force: bool,
) -> str:
    """Завести учётную запись и вернуть строку отчёта для печати."""
    vector_store = VectorStore(settings)
    try:
        # Коллекции создаются здесь же: первый оператор заводится до того, как
        # наполнение вообще запускалось, и падать на пустом хранилище ему не за чем.
        await vector_store.ensure_collections()
        users = build_user_repository(vector_store)
        hasher = build_password_hasher()
        clock = SystemClock()
        service = build_auth_service(
            users,
            build_refresh_session_repository(vector_store),
            hasher,
            build_access_token_issuer(settings, clock),
            build_refresh_token_factory(),
            clock,
            settings,
        )

        existing = await users.get_by_email(email)
        if existing is not None and not force:
            raise ApplicationError(
                f"Адрес {normalise_email(email)} уже занят. "
                "Повторите с --force, чтобы сменить пароль и роль этой записи."
            )
        if existing is not None:
            await reset_account(
                existing,
                users=users,
                hasher=hasher,
                password=password,
                role=role,
                display_name=display_name,
            )
            return f"Учётная запись {existing.email} обновлена: роль {role.value}."

        # Сессия здесь не открывается: скрипт заводит доступ, а не входит в него.
        user = await service.provision(
            email=email,
            password=password,
            display_name=display_name,
            role=role,
        )
        return f"Учётная запись {user.email} заведена: роль {user.role.value}."
    finally:
        await vector_store.close()


async def reset_account(
    user: User,
    *,
    users: UserRepositoryProtocol,
    hasher: PasswordHasherProtocol,
    password: str,
    role: UserRole,
    display_name: str | None,
) -> None:
    """Привести существующую запись к заданным роли, подписи и паролю."""
    user.password_hash = hasher.hash(password)
    user.role = role
    if display_name:
        user.display_name = display_name.strip()
    await users.update(user)


def main() -> None:
    """Точка входа скрипта."""
    arguments = parse_args()
    if len(arguments.email.strip()) < 3:
        raise SystemExit("Адрес почты слишком короткий.")

    settings = get_settings()
    if not settings.is_auth_configured:
        raise SystemExit(
            "AUTH_JWT_SECRET не задан: без ключа подписи терминал не выдаёт токены. "
            'Сгенерируйте его: python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )

    password = request_password(arguments.password)
    try:
        report = asyncio.run(
            create_account(
                settings,
                email=arguments.email,
                password=password,
                role=ROLES[arguments.role],
                display_name=arguments.name,
                force=arguments.force,
            )
        )
    except EmailAlreadyRegisteredError as error:
        raise SystemExit(str(error)) from error
    except ApplicationError as error:
        raise SystemExit(str(error)) from error

    print(report)


if __name__ == "__main__":
    main()
