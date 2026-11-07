from __future__ import annotations

from app.domain import User
from app.exceptions import AuthorizationError


class AccessPolicy:
    """Правила авторизации: кому и что позволено сделать с учётной записью.

    Правило про собственную запись выглядит мелочью, но именно на нём чаще всего
    и ломается система: администратор, снявший с себя роль или выключивший себя,
    оставляет терминал без единого оператора, и войти обратно можно только из CLI.
    """

    def ensure_admin(self, user: User) -> None:
        """Потребовать административных прав и поднять :class:`AuthorizationError`."""
        if not user.is_admin:
            raise AuthorizationError("Управление учётными записями доступно только администратору.")

    def ensure_can_manage(self, actor: User, target: User) -> None:
        """Проверить право администратора распоряжаться этой учётной записью."""
        self.ensure_admin(actor)
        if actor.identifier == target.identifier:
            raise AuthorizationError(
                "Администратор не вправе менять или удалять собственную учётную запись: "
                "так терминал остаётся без оператора."
            )
