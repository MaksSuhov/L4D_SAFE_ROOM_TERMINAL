from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm

from app.cookies import AuthCookies
from app.dependencies import (
    AuthCookiesDep,
    AuthServiceDep,
    CurrentUserDep,
    SettingsDep,
    require_csrf_token,
)
from app.exceptions import InvalidTokenError
from app.schemas import RefreshRequest, RegisterRequest, RevocationReport, SessionResponse, UserRead
from app.services import IssuedSession

router = APIRouter(prefix="/auth", tags=["auth"])

UNAUTHORIZED_RESPONSES: dict[int | str, dict[str, str]] = {
    401: {"description": "Учётные данные или токен не приняты"},
}


@router.post(
    "/register",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Завести учётную запись (и сразу войти)",
    responses={
        403: {"description": "Самостоятельная регистрация выключена"},
        409: {"description": "На эту почту уже заведена учётная запись"},
        422: {"description": "Адрес почты или пароль не отвечают требованиям"},
    },
)
async def register(
    payload: RegisterRequest,
    request: Request,
    response: Response,
    auth_service: AuthServiceDep,
    cookies: AuthCookiesDep,
) -> SessionResponse:
    """Открыть новый доступ к терминалу и выдать пару токенов."""
    issued = await auth_service.register(
        email=str(payload.email),
        password=payload.password,
        display_name=payload.display_name,
        user_agent=user_agent(request),
    )
    return establish_session(response, cookies, issued)


@router.post(
    "/token",
    response_model=SessionResponse,
    summary="Получить пару токенов (OAuth 2, password grant)",
    responses=UNAUTHORIZED_RESPONSES,
)
async def issue_tokens(
    request: Request,
    response: Response,
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    auth_service: AuthServiceDep,
    cookies: AuthCookiesDep,
) -> SessionResponse:
    """Обменять пару «почта — пароль» на токены.

    Форма полей здесь не украшение: это тот же контракт, что у кнопки
    «Authorize» в документации, поэтому адрес почты уезжает в поле ``username``.
    Токены возвращаются не в теле, а cookie — телу доверяет скрипт страницы, а
    cookie недоступна ему и потому не утекает вместе с XSS.
    """
    issued = await auth_service.log_in(
        email=form.username,
        password=form.password,
        user_agent=user_agent(request),
    )
    return establish_session(response, cookies, issued)


@router.post(
    "/refresh",
    response_model=SessionResponse,
    summary="Обновить пару токенов (OAuth 2, refresh grant)",
    responses={
        401: {
            "description": (
                "Refresh-токен не принят: неизвестен, истёк или предъявлен повторно — "
                "в последнем случае отзывается вся цепочка ротаций"
            )
        }
    },
)
async def refresh_tokens(
    request: Request,
    response: Response,
    auth_service: AuthServiceDep,
    cookies: AuthCookiesDep,
    settings: SettingsDep,
    payload: RefreshRequest | None = None,
) -> SessionResponse:
    """Обменять refresh-токен на новую пару и погасить предъявленный.

    Токен берётся из cookie, если он там есть, и только иначе — из тела: браузер
    прикладывает свой сам, а серверный клиент называет явно.
    """
    presented = request.cookies.get(settings.auth_refresh_cookie_name) or (
        payload.refresh_token if payload else None
    )
    if not presented:
        raise InvalidTokenError("Запрос не предъявил refresh-токен.")

    issued = await auth_service.refresh(presented, user_agent=user_agent(request))
    return establish_session(response, cookies, issued)


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Выйти на этом устройстве",
    dependencies=[Depends(require_csrf_token)],
)
async def log_out(
    request: Request,
    response: Response,
    auth_service: AuthServiceDep,
    cookies: AuthCookiesDep,
    settings: SettingsDep,
) -> None:
    """Погасить сессию и стереть cookie.

    Выход идемпотентен: повторный вызов и запрос без cookie дают тот же результат,
    потому что клиенту нечего из этого извлекать, а постороннему — тем более.
    """
    await auth_service.log_out(request.cookies.get(settings.auth_refresh_cookie_name))
    cookies.clear(response)


@router.post(
    "/logout-all",
    response_model=RevocationReport,
    summary="Выйти на всех устройствах",
    dependencies=[Depends(require_csrf_token)],
)
async def log_out_everywhere(
    response: Response,
    user: CurrentUserDep,
    auth_service: AuthServiceDep,
    cookies: AuthCookiesDep,
) -> RevocationReport:
    """Погасить все сессии учётной записи, включая текущую."""
    revoked = await auth_service.log_out_everywhere(user.identifier)
    cookies.clear(response)
    return RevocationReport(revoked=revoked)


@router.get(
    "/me",
    response_model=UserRead,
    summary="Кто представлен текущей сессией",
    responses=UNAUTHORIZED_RESPONSES,
)
async def read_current_user(user: CurrentUserDep) -> UserRead:
    """Рассказать клиенту, от чьего имени он обращается.

    По этому запросу страница после перезагрузки узнаёт, что сессия ещё жива, —
    токены ей для этого читать не нужно и нельзя.
    """
    return UserRead.model_validate(user)


def establish_session(
    response: Response,
    cookies: AuthCookies,
    issued: IssuedSession,
) -> SessionResponse:
    """Разложить выданную сессию по cookie и собрать тело ответа."""
    cookies.set_session(
        response,
        access_token=issued.access_token.value,
        refresh_token=issued.refresh_token,
        csrf_token=issued.csrf_token,
        access_max_age=issued.expires_in,
        refresh_max_age=issued.refresh_expires_in,
    )
    return SessionResponse(
        user=UserRead.model_validate(issued.user),
        csrf_token=issued.csrf_token,
        expires_in=issued.expires_in,
    )


def user_agent(request: Request) -> str:
    """Кто именно держит сессию: записывается ради разбора, а не ради проверки."""
    return request.headers.get("user-agent", "")
