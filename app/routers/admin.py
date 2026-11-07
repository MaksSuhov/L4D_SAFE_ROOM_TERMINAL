from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.dependencies import (
    AccessPolicyDep,
    AdminUserDep,
    AuthServiceDep,
    UserRepositoryDep,
    require_csrf_token,
)
from app.exceptions import ResourceNotFoundError
from app.schemas import RevocationReport, UserRead, UserUpdate

router = APIRouter(prefix="/admin", tags=["admin"])

ADMIN_RESPONSES: dict[int | str, dict[str, str]] = {
    401: {"description": "Запрос не предъявил действующую сессию"},
    403: {"description": "Учётная запись не администратор или запрос без CSRF-метки"},
}


@router.get(
    "/users",
    response_model=list[UserRead],
    summary="Список учётных записей терминала",
    responses=ADMIN_RESPONSES,
)
async def list_users(admin: AdminUserDep, users: UserRepositoryDep) -> list[UserRead]:
    """Вернуть все учётные записи, упорядоченные по адресу почты."""
    return [UserRead.model_validate(user) for user in await users.get_all()]


@router.patch(
    "/users/{user_id}",
    response_model=UserRead,
    summary="Изменить роль или доступ учётной записи",
    responses={**ADMIN_RESPONSES, 404: {"description": "Такой учётной записи нет"}},
    dependencies=[Depends(require_csrf_token)],
)
async def update_user(
    user_id: str,
    payload: UserUpdate,
    admin: AdminUserDep,
    users: UserRepositoryDep,
    policy: AccessPolicyDep,
) -> UserRead:
    """Сменить роль или заблокировать учётную запись.

    Блокировка действует сразу: и токен доступа, и refresh-токен проверяются
    против записи, поэтому погашать сессии отдельно не нужно — роль и признак
    активности читаются оттуда, а не из выданного ранее токена.
    """
    target = await users.get_by_id(user_id)
    if target is None:
        raise ResourceNotFoundError("User", user_id)

    policy.ensure_can_manage(admin, target)
    if payload.role is not None:
        target.role = payload.role
    if payload.is_active is not None:
        target.is_active = payload.is_active
    await users.update(target)
    return UserRead.model_validate(target)


@router.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить учётную запись",
    responses={**ADMIN_RESPONSES, 404: {"description": "Такой учётной записи нет"}},
    dependencies=[Depends(require_csrf_token)],
)
async def delete_user(
    user_id: str,
    admin: AdminUserDep,
    users: UserRepositoryDep,
    policy: AccessPolicyDep,
    auth_service: AuthServiceDep,
) -> None:
    """Удалить учётную запись вместе с её сессиями."""
    target = await users.get_by_id(user_id)
    if target is None:
        raise ResourceNotFoundError("User", user_id)

    policy.ensure_can_manage(admin, target)
    # Сессии гасятся первыми и намеренно: без записи они уже не работают, но
    # оставлять их в хранилище означало бы держать следы доступа удалённого человека.
    await auth_service.log_out_everywhere(user_id)
    await users.delete(user_id)


@router.post(
    "/users/{user_id}/sessions/revoke",
    response_model=RevocationReport,
    summary="Разлогинить учётную запись на всех устройствах",
    responses={**ADMIN_RESPONSES, 404: {"description": "Такой учётной записи нет"}},
    dependencies=[Depends(require_csrf_token)],
)
async def revoke_user_sessions(
    user_id: str,
    admin: AdminUserDep,
    users: UserRepositoryDep,
    auth_service: AuthServiceDep,
) -> RevocationReport:
    """Погасить все refresh-сессии учётной записи.

    Выданный токен доступа при этом доживает свой срок: отозвать его нельзя, и
    именно поэтому он живёт минуты, а не месяцы.
    """
    if await users.get_by_id(user_id) is None:
        raise ResourceNotFoundError("User", user_id)

    return RevocationReport(revoked=await auth_service.log_out_everywhere(user_id))
