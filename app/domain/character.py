"""Доменная сущность образа выжившего."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Character:
    """Образ выжившего, в которого терминал умеет перевоплощаться."""

    name: str
    system_prompt: str
    few_shot_examples: list[dict[str, str]] = field(default_factory=list)
    id: int | None = None
    """Идентификатор образа; до сохранения его нет, его выдаёт хранилище."""
