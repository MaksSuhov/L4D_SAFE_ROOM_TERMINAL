"""Renders retrieved evidence as the two-level archive brief handed to the model.

This module owns the *policy* of the pipeline — how a fact differs from a rumour
and how the model is allowed to use each. Keeping it here, instead of in the
provider adapter, means the wording can be tested without any network access and
changed without touching the transport layer.

The brief has three blocks:

``<official_lore>``
    Verified encyclopaedia material. The only place a character may take a
    statement of fact from.
``<wall_graffiti>``
    Messages left by other survivors on safe-room walls. Hearsay, attributed
    aloud ("на стене написано…") and coloured by the character's temperament —
    never presented as confirmed lore.
``<speech_samples>``
    Lines actually spoken by the answering character, used as a style reference
    so the answer sounds like them rather than like a narrator.
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
    """The three tiers retrieved for one question."""

    official: Sequence[RetrievedChunk] = ()
    graffiti: Sequence[RetrievedChunk] = ()
    voice: Sequence[RetrievedChunk] = ()

    @property
    def is_empty(self) -> bool:
        """Whether no tier produced a record at all."""
        return not (self.official or self.graffiti or self.voice)

    @property
    def all_chunks(self) -> list[RetrievedChunk]:
        """Every retrieved record, official lore first, for the API response."""
        return [*self.official, *self.graffiti, *self.voice]


class ArchivePromptBuilder:
    """Builds the evidence message of a chat turn from :class:`ArchiveEvidence`.

    The builder is stateless: the persona-specific part of the prompt is the
    character's own system prompt, so the same instance serves every request.
    """

    def build(self, evidence: ArchiveEvidence) -> str:
        """Render ``evidence`` as the multi-block archive brief."""
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
        """Render one numbered evidence block, or its placeholder when empty."""
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
        """Render the character's own lines as a quoted style reference.

        A voice sample holds several recorded lines, one per line of text, and
        each becomes its own bullet so the model sees them as separate quotes
        rather than as one long block.
        """
        rendered: list[str] = ["", f"<{VOICE_SECTION_HEADER}>"]
        for match in matches:
            lines = [line for line in match.chunk.content.splitlines() if line.strip()]
            rendered.extend(f"— {line}" for line in lines)
        rendered.append(f"</{VOICE_SECTION_HEADER}>")
        return rendered
