from __future__ import annotations

import logging
import re

from app.exceptions import AIServiceError
from app.interfaces import AIClientProtocol, ExpandedQuery, QueryExpanderProtocol, TracerProtocol
from app.services.json_array import extract_json_array
from app.services.tracing import NoOpTracer

logger = logging.getLogger("l4d.hide")

MIN_WORDS_FOR_EXPANSION = 3
"""Короче этого вопрос не разворачивается: в «привет» нечего раскладывать на сущности,
а расширение стоит запроса к провайдеру на каждом ходу разговора.
"""

ITEM_SEPARATORS = re.compile(r"[,\n;]")
"""Разделители, по которым строки массива режутся, если модель забыла кавычки."""

QUOTE_PAIRS: dict[str, str] = {
    '"': '"',
    "'": "'",
    "«": "»",
    "“": "”",
    "„": "“",
    "‘": "’",
}
"""Парные кавычки, которые модель может добавить вокруг формулировки или вместо них."""


def strip_outer_quotes(text: str) -> str:
    """Снять парные кавычки вокруг формулировки, не трогая кавычки внутри неё."""
    while len(text) >= 2 and QUOTE_PAIRS.get(text[0]) == text[-1]:
        text = text[1:-1].strip()
    return text

HIDE_SYSTEM_PROMPT: str = (
    "Ты — поисковый аналитик архива Left 4 Dead. Тебе задают вопрос игрока, и твоя "
    "задача — не отвечать на него, а построить формулировки, по которым архив найдёт "
    "нужные записи.\n\n"
    "Как строить формулировки:\n"
    "- Разложи вопрос на сущности и связи между ними: персонаж, кампания, карта, "
    "оружие, предмет, заражённый, режим. На каждую сущность и на каждую связь — "
    "отдельная формулировка.\n"
    "- Называй сущности так, как они называются в русской вики (Джимми Гиббс-младший, "
    "«Вымерший центр», «Атриум»), а не словами вопроса.\n"
    "- Пиши короткие вопросы и заголовки, а не рассуждения.\n"
    "- Не додумывай связей, которых не знаешь: лишняя уверенная гипотеза хуже "
    "недостающей.\n"
    "- Последняя формулировка — гипотетический фрагмент статьи, который лежал бы в "
    "архиве, если бы ответ на вопрос существовал: одно-два предложения "
    "энциклопедическим тоном, с теми же терминами.\n\n"
    "Пример. Вопрос: «какое оружие есть в кампании с Джимми Гиббсом-младшим?» "
    'Формулировки: ["Джимми Гиббс-младший — необычный заражённый из какой кампании?", '
    '"Оружие в кампании «Вымерший центр»", "Что находят выжившие на картах «Вымершего '
    'центра»?", "Джимми Гиббс-младший — редкий заражённый кампании «Вымерший центр»: '
    'его можно встретить в главе «Атриум», где выжившие находят дробовики, штурмовые '
    'винтовки и гранаты."]\n\n'
    "Отвечай ТОЛЬКО JSON-массивом строк, не длиннее {max_probes}, без markdown, "
    "нумерации и пояснений вокруг."
)


class HideQueryExpander(QueryExpanderProtocol):
    """HIDE: разворачивает вопрос в гипотетические формулировки и фрагмент статьи.

    Игрок спрашивает про Джимми Гиббса-младшего, а запись про оружие лежит в статье
    о кампании «Вымерший центр»: общего слова у вопроса и записи нет, и поиск по
    одному вопросу до этой статьи не доходит. Расширитель просит модель разложить
    вопрос на сущности и связи, назвать их так, как они названы в вики, и дописать
    гипотетический фрагмент статьи — каждая формулировка ищется отдельно, так что
    мост «персонаж → кампания → карта → оружие» проходится по шагам.

    Расширение никогда не срывает ответ: отказ провайдера или неразобранный ответ
    оставляют поиск по исходному вопросу.
    """

    def __init__(
        self,
        ai_client: AIClientProtocol,
        *,
        max_questions: int = 4,
        tracer: TracerProtocol | None = None,
    ) -> None:
        self._ai_client = ai_client
        self._max_questions = max_questions
        self._tracer = tracer or NoOpTracer()
        self._system_prompt = HIDE_SYSTEM_PROMPT.format(max_probes=max_questions)

    async def expand(self, question: str) -> ExpandedQuery:
        """Развернуть ``question`` в формулировки, по которым пойдёт поиск."""
        if self._max_questions <= 0 or len(question.split()) < MIN_WORDS_FOR_EXPANSION:
            return ExpandedQuery(original=question)

        with self._tracer.span("hide-expand") as span:
            try:
                reply = await self._ai_client.complete(
                    self._system_prompt,
                    question,
                    temperature=0.0,
                )
                hypotheticals = self._parse(question, reply)
            except AIServiceError as error:
                logger.warning("HIDE expansion failed, searching the question as asked: %s", error)
                span.annotate(input=question, output=())
                return ExpandedQuery(original=question)

            span.annotate(input=question, output=hypotheticals)

        logger.debug("HIDE extended the question with %d probes", len(hypotheticals))
        return ExpandedQuery(original=question, hypotheticals=hypotheticals)

    def _parse(self, question: str, reply: str) -> tuple[str, ...]:
        """Отобрать формулировки: без пустых, без повтора вопроса и не длиннее лимита."""
        asked = question.strip().casefold()
        picked: list[str] = []
        for item in self._read_items(reply):
            candidate = strip_outer_quotes(item.strip())
            if not candidate or candidate.casefold() == asked or candidate in picked:
                continue
            picked.append(candidate)
            if len(picked) == self._max_questions:
                break
        return tuple(picked)

    @staticmethod
    def _read_items(reply: str) -> list[str]:
        """Прочитать строки массива: сначала строгим JSON, при неудаче — по разделителям.

        Разделители выручают там, где модель перечислила формулировки без кавычек;
        ответ вовсе без массива остаётся ошибкой, и вопрос ищется так, как задан.
        """
        try:
            return extract_json_array(reply)
        except AIServiceError:
            if "[" not in reply or "]" not in reply:
                raise
            body = reply[reply.index("[") + 1 : reply.rindex("]")]
        return ITEM_SEPARATORS.split(body)


class NoQueryExpander(QueryExpanderProtocol):
    """Выключенный HIDE: поиск идёт по вопросу игрока и не стоит ни одного запроса."""

    async def expand(self, question: str) -> ExpandedQuery:
        """Вернуть вопрос без гипотетических формулировок."""
        return ExpandedQuery(original=question)


__all__ = [
    "MIN_WORDS_FOR_EXPANSION",
    "HIDE_SYSTEM_PROMPT",
    "HideQueryExpander",
    "NoQueryExpander",
]
