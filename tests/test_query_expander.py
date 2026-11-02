"""Модульные тесты расширения вопроса гипотетическими формулировками (HIDE)."""

from __future__ import annotations

import json

from app.exceptions import AIServiceError
from app.interfaces import ExpandedQuery
from app.services.query_expander import HideQueryExpander, NoQueryExpander
from tests.fakes import FakeAIClient

QUESTION = "какое оружие есть в компании с джимми гиббсом младшим?"

HYPOTHETICALS = [
    "Джимми Гиббс-младший — машина из какой кампании?",
    "Оружие в кампании «Вымерший центр»",
    "Джимми Гиббс-младший стоит в «Атриуме» — последней карте «Вымершего центра».",
]


class FailingAIClient(FakeAIClient):
    """Заглушка, чей ответ на одну инструкцию всегда падает."""

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.0,
    ) -> str:
        raise AIServiceError("provider is down")


def build_expander(
    reply: str = "[]",
    *,
    max_questions: int = 4,
) -> tuple[HideQueryExpander, FakeAIClient]:
    """Собрать расширитель над заглушкой клиента с заданным ответом модели."""
    client = FakeAIClient(completion=reply)
    return HideQueryExpander(client, max_questions=max_questions), client


async def test_the_original_question_leads_the_probes() -> None:
    expander, _ = build_expander(json.dumps(HYPOTHETICALS, ensure_ascii=False))

    expanded = await expander.expand(QUESTION)

    assert expanded.original == QUESTION
    assert expanded.hypotheticals == tuple(HYPOTHETICALS)
    assert expanded.probes == (QUESTION, *HYPOTHETICALS)
    assert expanded.is_expanded


async def test_the_question_is_asked_verbatim_and_deterministically() -> None:
    expander, client = build_expander("[]")

    await expander.expand(QUESTION)

    call = client.completion_calls[0]
    assert call["user_message"] == QUESTION
    assert call["temperature"] == 0.0
    assert "JSON" in str(call["system_prompt"])


async def test_the_limit_on_formulations_reaches_the_prompt() -> None:
    expander, client = build_expander("[]", max_questions=2)

    await expander.expand(QUESTION)

    assert "не длиннее 2" in str(client.completion_calls[0]["system_prompt"])


async def test_the_hypotheticals_are_capped_by_the_limit() -> None:
    expander, _ = build_expander(json.dumps(HYPOTHETICALS, ensure_ascii=False), max_questions=2)

    expanded = await expander.expand(QUESTION)

    assert expanded.hypotheticals == tuple(HYPOTHETICALS[:2])


async def test_blank_repeats_of_the_question_and_duplicates_are_dropped() -> None:
    reply = json.dumps(
        [QUESTION, "", "   ", HYPOTHETICALS[0], HYPOTHETICALS[0], HYPOTHETICALS[1]],
        ensure_ascii=False,
    )
    expander, _ = build_expander(reply)

    expanded = await expander.expand(QUESTION)

    assert expanded.hypotheticals == (HYPOTHETICALS[0], HYPOTHETICALS[1])


async def test_a_reply_that_is_not_a_json_array_keeps_the_question_alone() -> None:
    expander, _ = build_expander("Извини, не могу помочь с этим вопросом.")

    expanded = await expander.expand(QUESTION)

    assert expanded == ExpandedQuery(original=QUESTION)
    assert expanded.probes == (QUESTION,)


async def test_an_array_without_quotes_is_still_read() -> None:
    """Проверить, что массив без кавычек не выбрасывается целиком: провайдер изредка
    возвращает именно такой ответ, и мост по нему строится тот же.
    """
    expander, _ = build_expander("[Оружие в «Вымершем центре»,\nЧто лежит в «Атриуме»?]")

    expanded = await expander.expand(QUESTION)

    assert expanded.hypotheticals == ("Оружие в «Вымершем центре»", "Что лежит в «Атриуме»?")


async def test_quoted_formulations_lose_their_outer_quotes() -> None:
    expander, _ = build_expander(json.dumps(["«Первая»", "Вторая"], ensure_ascii=False))

    expanded = await expander.expand(QUESTION)

    assert expanded.hypotheticals == ("Первая", "Вторая")


async def test_a_provider_failure_keeps_the_question_alone() -> None:
    expander = HideQueryExpander(FailingAIClient(), max_questions=4)

    expanded = await expander.expand(QUESTION)

    assert expanded.probes == (QUESTION,)


async def test_a_short_question_is_not_expanded_at_all() -> None:
    expander, client = build_expander(json.dumps(HYPOTHETICALS, ensure_ascii=False))

    expanded = await expander.expand("Привет")

    assert expanded.probes == ("Привет",)
    assert client.completion_calls == []


async def test_a_zero_limit_is_not_expanded_at_all() -> None:
    reply = json.dumps(HYPOTHETICALS, ensure_ascii=False)
    expander, client = build_expander(reply, max_questions=0)

    expanded = await expander.expand(QUESTION)

    assert expanded.probes == (QUESTION,)
    assert client.completion_calls == []


async def test_the_disabled_expander_searches_the_question_as_asked() -> None:
    expanded = await NoQueryExpander().expand(QUESTION)

    assert expanded == ExpandedQuery(original=QUESTION)
    assert not expanded.is_expanded
