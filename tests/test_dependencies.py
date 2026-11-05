"""Проверки сборки зависимостей: настройки решают, чем окажется расширитель."""

from __future__ import annotations

import json

from app.config import Settings
from app.dependencies import build_query_expander
from app.services import HideQueryExpander, NoOpTracer, NoQueryExpander, build_tracer
from tests.fakes import FakeAIClient

QUESTION = "какое оружие есть в компании с джимми гиббсом младшим?"


def test_the_enabled_setting_builds_the_hide_expander() -> None:
    expander = build_query_expander(
        FakeAIClient(),
        Settings(ai_api_key="k", hide_enabled=True),
        NoOpTracer(),
    )

    assert isinstance(expander, HideQueryExpander)


def test_the_disabled_setting_builds_the_null_expander() -> None:
    expander = build_query_expander(
        FakeAIClient(),
        Settings(ai_api_key="k", hide_enabled=False),
        NoOpTracer(),
    )

    assert isinstance(expander, NoQueryExpander)


async def test_the_configured_limit_reaches_the_expander() -> None:
    reply = json.dumps(["Первая", "Вторая", "Третья", "Четвёртая"], ensure_ascii=False)
    client = FakeAIClient(completion=reply)
    expander = build_query_expander(
        client,
        Settings(ai_api_key="k", hide_enabled=True, hide_max_questions=1),
        NoOpTracer(),
    )

    expanded = await expander.expand(QUESTION)

    assert expanded.hypotheticals == ("Первая",)


def test_without_keys_the_tracer_is_the_null_one() -> None:
    settings = Settings(ai_api_key="k", langfuse_public_key="", langfuse_secret_key="")

    assert settings.is_langfuse_configured is False
    assert isinstance(build_tracer(settings), NoOpTracer)


def test_the_switch_beats_the_keys() -> None:
    settings = Settings(
        ai_api_key="k",
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
        langfuse_enabled=False,
    )

    assert settings.is_langfuse_configured is True
    assert isinstance(build_tracer(settings), NoOpTracer)
