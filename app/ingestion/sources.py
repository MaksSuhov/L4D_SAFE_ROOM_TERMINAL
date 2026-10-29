"""Declarative registry of the wiki pages that make up the corpus.

Every page the pipeline reads is declared here once, together with the topic it
belongs to and the parser that knows its layout. Adding a page to the corpus is
therefore a one-line change, and the set of sources is reviewable at a glance
instead of being scattered across the code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.taxonomy import LoreTopic, SourceType


class ParserKind(StrEnum):
    """Layout families the wiki uses, each handled by its own parser."""

    GRAFFITI = "graffiti"
    """Campaign › chapter › list of wall inscriptions."""

    ARTICLE = "article"
    """Encyclopaedia prose with stat tables, split by heading."""

    QUOTES = "quotes"
    """Dialogue lists grouped by situation."""

    @property
    def source_type(self) -> SourceType:
        """Evidence tier produced by this layout.

        Deriving the tier from the layout keeps the two in step: a quotes page
        can only ever produce voice samples, and an article page can only ever
        produce official lore.
        """
        return {
            ParserKind.GRAFFITI: SourceType.GRAFFITI,
            ParserKind.ARTICLE: SourceType.OFFICIAL,
            ParserKind.QUOTES: SourceType.VOICE,
        }[self]


@dataclass(frozen=True, slots=True)
class WikiSource:
    """One wiki page of the corpus."""

    page: str
    topic: LoreTopic
    parser: ParserKind
    character_name: str | None = None
    """Character the page belongs to (their quotes) or is about (their biography)."""


BILL = "Билл"
ZOEY = "Зои"
FRANCIS = "Фрэнсис"
LOUIS = "Луис"
COACH = "Тренер"
NICK = "Ник"
ROCHELLE = "Рошель"
ELLIS = "Эллис"

CHARACTER_PAGES: tuple[tuple[str, str], ...] = (
    ("Bill", BILL),
    ("Zoey", ZOEY),
    ("Francis", FRANCIS),
    ("Louis", LOUIS),
    ("Coach", COACH),
    ("Nick", NICK),
    ("Rochelle", ROCHELLE),
    ("Ellis", ELLIS),
)
"""Wiki article title and the Russian display name stored in the database."""

QUOTE_PAGES: tuple[tuple[str, str], ...] = (
    ("Bill/Quotes and Captions", BILL),
    ("Zoey/Quotes and Captions", ZOEY),
    ("Francis/Quotes and Captions", FRANCIS),
    ("Louis/Quotes and Captions", LOUIS),
    ("Coach/Quotes", COACH),
    ("Nick/Quotes", NICK),
    ("Rochelle/Quotes", ROCHELLE),
    ("Ellis/Quotes", ELLIS),
)
"""Where the full dialogue dumps of each survivor live."""

INFECTION_PAGES: tuple[str, ...] = (
    "Green Flu",
    "Carriers",
    "CEDA",
    "Military",
    "Media Coverage",
)

INFECTED_PAGES: tuple[str, ...] = (
    "Common Infected",
    "Uncommon Infected",
    "Special Infected",
    "The Tank",
    "The Witch",
    "Boomer",
    "Hunter",
    "Smoker",
    "Charger",
    "Spitter",
    "Jockey",
)
"""The infected pages that carry the facts about the mutations.

Two traps live in this list, both found by reading the wiki rather than the code.
``Tank`` is a disambiguation stub — the statistics are on ``The Tank`` — and
``Witch``, ``The Witch`` and ``Wandering Witch`` are three titles of one page, so
only one of them is declared here. ``The Infected`` is likewise a redirect of
``Special Infected``.
"""

CAMPAIGN_PAGES: tuple[str, ...] = (
    "Campaigns",
    "No Mercy",
    "Crash Course",
    "Death Toll",
    "Dead Air",
    "Blood Harvest",
    "The Last Stand (Left 4 Dead)",
    "The Sacrifice",
    "Dead Center",
    "The Passing",
    "Dark Carnival",
    "Swamp Fever",
    "Hard Rain",
    "The Parish",
    "Cold Stream",
)

GRAFFITI_PAGE = "Graffiti"
"""Holds the inscriptions of every safe room, grouped by campaign and chapter."""

SAFE_ROOM_PAGES: tuple[str, ...] = ("Safe room",)


def _sources() -> tuple[WikiSource, ...]:
    """Assemble the registry, keeping the declaration order as the fetch order."""
    sources: list[WikiSource] = [
        WikiSource(page, LoreTopic.INFECTION, ParserKind.ARTICLE) for page in INFECTION_PAGES
    ]
    sources.append(WikiSource("Survivors", LoreTopic.SURVIVORS, ParserKind.ARTICLE))
    sources.extend(
        WikiSource(page, LoreTopic.SURVIVORS, ParserKind.ARTICLE, character_name=character)
        for page, character in CHARACTER_PAGES
    )
    sources.extend(
        WikiSource(page, LoreTopic.INFECTED, ParserKind.ARTICLE) for page in INFECTED_PAGES
    )
    sources.extend(
        WikiSource(page, LoreTopic.CAMPAIGNS, ParserKind.ARTICLE) for page in CAMPAIGN_PAGES
    )
    sources.extend(
        WikiSource(page, LoreTopic.SAFE_ROOMS, ParserKind.ARTICLE) for page in SAFE_ROOM_PAGES
    )
    sources.append(WikiSource(GRAFFITI_PAGE, LoreTopic.GRAFFITI, ParserKind.GRAFFITI))
    sources.extend(
        WikiSource(page, LoreTopic.QUOTES, ParserKind.QUOTES, character_name=character)
        for page, character in QUOTE_PAGES
    )
    return tuple(sources)


WIKI_SOURCES: Sequence[WikiSource] = _sources()
"""Every page the corpus is built from, in fetch order."""
