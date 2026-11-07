from __future__ import annotations

from fastapi import Response

from app.config import Settings


class AuthCookies:
    """Раскладка токенов по cookie: одни и те же имена и флаги на входе и на выходе.

    Обе части пары живут в cookie, недоступных скриптам страницы, поэтому XSS не
    уносит сессию вместе с содержимым страницы. CSRF-метка, наоборот, читается из
    скриптов намеренно: клиент обязан повторить её заголовком, и в этом весь смысл
    двойной отправки.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def set_session(
        self,
        response: Response,
        *,
        access_token: str,
        refresh_token: str,
        csrf_token: str,
        access_max_age: int,
        refresh_max_age: int,
    ) -> None:
        """Разложить пару токенов и CSRF-метку по cookie ответа."""
        self._write(
            response,
            self._settings.auth_access_cookie_name,
            access_token,
            "/",
            access_max_age,
            http_only=True,
        )
        self._write(
            response,
            self._settings.auth_refresh_cookie_name,
            refresh_token,
            self._settings.auth_refresh_cookie_path,
            refresh_max_age,
            http_only=True,
        )
        self._write(
            response,
            self._settings.auth_csrf_cookie_name,
            csrf_token,
            "/",
            refresh_max_age,
            http_only=False,
        )

    def clear(self, response: Response) -> None:
        """Стереть все cookie сессии: выход не должен оставлять следов в браузере."""
        for name, path in self._cookie_layout():
            response.delete_cookie(
                key=name,
                path=path,
                domain=self._settings.auth_cookie_domain or None,
                secure=self._settings.auth_cookie_secure,
                samesite=self._settings.auth_cookie_samesite,
            )

    def _write(
        self,
        response: Response,
        name: str,
        value: str,
        path: str,
        max_age: int,
        *,
        http_only: bool,
    ) -> None:
        """Записать одну cookie с общими для всех тремя флагами безопасности."""
        response.set_cookie(
            key=name,
            value=value,
            max_age=max_age,
            path=path,
            domain=self._settings.auth_cookie_domain or None,
            httponly=http_only,
            secure=self._settings.auth_cookie_secure,
            samesite=self._settings.auth_cookie_samesite,
        )

    def _cookie_layout(self) -> tuple[tuple[str, str], ...]:
        """Имена и пути всех cookie сессии — в одном месте, чтобы их не разъехались."""
        return (
            (self._settings.auth_access_cookie_name, "/"),
            (self._settings.auth_refresh_cookie_name, self._settings.auth_refresh_cookie_path),
            (self._settings.auth_csrf_cookie_name, "/"),
        )
