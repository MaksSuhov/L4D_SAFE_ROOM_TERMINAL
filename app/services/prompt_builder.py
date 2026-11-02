"""Оформляет найденные свидетельства как двухуровневую архивную сводку для модели.

Модуль отвечает за *политику* конвейера: чем факт отличается от слуха и как
модели разрешено пользоваться каждым из них. Держать её здесь, а не в адаптере
провайдера, — значит, что формулировки можно проверить без обращения к сети и
поменять, не трогая транспортный слой.

В сводке три блока:

``<official_lore>``
    Проверенный энциклопедический материал. Единственный источник, из которого
    персонаж вправе брать утверждение факта.
``<wall_graffiti>``
    Сообщения, оставленные другими выжившими на стенах убежищ. Слухи, которые
    пересказываются вслух («на стене написано…») и окрашиваются характером
    персонажа — никогда не подаются как подтверждённый лор.
``<speech_samples>``
    Реплики, реально произнесённые отвечающим персонажем, — образец стиля,
    чтобы ответ звучал как он, а не как рассказчик.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.interfaces import RetrievedChunk

ARCHIVE_INTRO: str = (
    "Ниже — выдержки из полевого архива выживших. Они разделены на два уровня "
    "достоверности, и путать их нельзя."
)

OFFICIAL_TIER_INSTRUCTION: str = (
    "1. Блок «official_lore» — проверенные сведения об игре. Только на них можно "
    "ссылаться как на факт, и только они дают право что-то утверждать. Если "
    "вопроса там нет, честно скажи, что в архиве об этом ничего нет, и не "
    "выдумывай механики, цифры и детали."
)

GRAFFITI_TIER_INSTRUCTION: str = (
    "2. Блок «wall_graffiti» — надписи, которые другие выжившие оставили на стенах "
    "убежищ. Это слухи, домыслы и чужие слова, а не факты. Пересказывай их "
    "только как чужую надпись, вслух указывая, что это написано на стене, и "
    "относись к ним так, как свойственно твоему характеру: кто-то язвит, кто-то "
    "пугается, кто-то принимает за чистую монету. Никогда не подавай граффити "
    "как официально подтверждённую правду."
)

VOICE_TIER_INSTRUCTION: str = (
    "3. Блок «speech_samples» — реальные реплики твоего персонажа из игры. Это "
    "образец твоей манеры речи: лексика, тон, любимые словечки и темы. Копируй "
    "манеру, но не пересказывай сами реплики как события."
)

STYLE_INSTRUCTION: str = (
    "Отвечай в образе, опираясь на эти материалы. Не упоминай, что ты языковая "
    "модель, и не ссылайся на «записи архива» как на документ — говори о них "
    "так, как говорил бы выживший."
)

OFFICIAL_SECTION_HEADER: str = "official_lore"
GRAFFITI_SECTION_HEADER: str = "wall_graffiti"
VOICE_SECTION_HEADER: str = "speech_samples"

NO_OFFICIAL_LORE: str = "(официальных сведений по этому вопросу в архиве нет)"
NO_GRAFFITI: str = "(подходящих надписей на стенах не найдено)"


@dataclass(frozen=True, slots=True)
class ArchiveEvidence:
    """Три уровня, найденные для одного вопроса."""

    official: Sequence[RetrievedChunk] = ()
    graffiti: Sequence[RetrievedChunk] = ()
    voice: Sequence[RetrievedChunk] = ()

    @property
    def is_empty(self) -> bool:
        """Правда ли, что ни один уровень не дал ни одной записи."""
        return not (self.official or self.graffiti or self.voice)

    @property
    def all_chunks(self) -> list[RetrievedChunk]:
        """Все найденные записи, официальный лор первым, для ответа API."""
        return [*self.official, *self.graffiti, *self.voice]


class ArchivePromptBuilder:
    """Собирает сообщение с доказательствами для одного хода разговора из :class:`ArchiveEvidence`.

    Конструктор не хранит состояния: часть промпта, зависящая от образа, — это
    собственный системный промпт персонажа, поэтому один и тот же экземпляр
    обслуживает любой запрос.
    """

    def build(self, evidence: ArchiveEvidence) -> str:
        """Оформить ``evidence`` как многочастную архивную сводку."""
        if evidence.is_empty:
            return ""

        lines: list[str] = [
            ARCHIVE_INTRO,
            "",
            OFFICIAL_TIER_INSTRUCTION,
            GRAFFITI_TIER_INSTRUCTION,
            VOICE_TIER_INSTRUCTION,
            "",
            STYLE_INSTRUCTION,
        ]
        lines.extend(
            self._render_tier(OFFICIAL_SECTION_HEADER, evidence.official, NO_OFFICIAL_LORE)
        )
        lines.extend(self._render_tier(GRAFFITI_SECTION_HEADER, evidence.graffiti, NO_GRAFFITI))
        lines.extend(self._render_voice(evidence.voice))
        return "\n".join(lines)

    @staticmethod
    def _render_tier(
        header: str,
        matches: Sequence[RetrievedChunk],
        placeholder: str,
    ) -> list[str]:
        """Оформить один нумерованный блок доказательств или заглушку, когда он пуст."""
        rendered: list[str] = ["", f"<{header}>"]
        if matches:
            rendered.extend(
                f"{position}. [{match.chunk.title}] {match.chunk.content}"
                for position, match in enumerate(matches, start=1)
            )
        else:
            rendered.append(placeholder)
        rendered.append(f"</{header}>")
        return rendered

    @staticmethod
    def _render_voice(matches: Sequence[RetrievedChunk]) -> list[str]:
        """Оформить собственные реплики персонажа как цитируемый образец стиля.

        Голосовой образец хранит несколько записанных реплик, по одной на строку
        текста, и каждая становится отдельным пунктом, чтобы модель видела их как
        отдельные цитаты, а не как один длинный блок.
        """
        rendered: list[str] = ["", f"<{VOICE_SECTION_HEADER}>"]
        for match in matches:
            lines = [line for line in match.chunk.content.splitlines() if line.strip()]
            rendered.extend(f"— {line}" for line in lines)
        rendered.append(f"</{VOICE_SECTION_HEADER}>")
        return rendered
