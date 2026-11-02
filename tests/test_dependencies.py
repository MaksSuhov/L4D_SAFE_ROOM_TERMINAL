"""Проверки сборки зависимостей: настройки решают, чем окажется расширитель."""

from __future__ import annotations

import json

from app.config import Settings
from app.dependencies import build_query_expander
from app.services import HideQueryExpander, NoQueryExpander
from tests.fakes import FakeAIClient

QUESTION = "какое оружие есть в компании с джимми гиббсом младшим?"


def test_the_enabled_setting_builds_the_hide_expander() -> None:
    expander = build_query_expander(FakeAIClient(), Settings(ai_api_key="k", hide_enabled=True))

    assert isinstance(expander, HideQueryExpander)


def test_the_disabled_setting_builds_the_null_expander() -> None:
    expander = build_query_expander(FakeAIClient(), Settings(ai_api_key="k", hide_enabled=False))

    assert isinstance(expander, NoQueryExpander)


async def test_the_configured_limit_reaches_the_expander() -> None:
    reply = json.dumps(["Первая", "Вторая", "Третья", "Четвёртая"], ensure_ascii=False)
    client = FakeAIClient(completion=reply)
    expander = build_query_expander(
        client,
        Settings(ai_api_key="k", hide_enabled=True, hide_max_questions=1),
    )

    expanded = await expander.expand(QUESTION)

    assert expanded.hypotheticals == ("Первая",)
