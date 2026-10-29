"""Parsers turning one wiki page into corpus records.

Each parser owns the layout of one :class:`~app.ingestion.sources.ParserKind`
and produces :class:`~app.schemas.LoreChunkCreate` records — flat, self-contained
text with enough provenance to be shown to a player. Splitting is deliberate:

* graffiti becomes one record per thread, because the walls hold conversations
  and a reply is meaningless without the inscription it answers;
* articles become one record per paragraph group of a section, so a retrieved
  chunk is small enough to fit several of them into the prompt;
* quotes become one record per situation of one character, which is the unit a
  voice sample is useful in.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from app.ingestion.sources import WikiSource
from app.ingestion.wikitext import ListItem, Section, WikitextCleaner
from app.schemas import LoreChunkCreate

ARTICLE_MAX_CHARS = 900
"""Upper bound of an official record; roughly one dense paragraph."""

ARTICLE_MIN_CHARS = 120
"""Shorter sections are merged into the neighbouring record instead of standing alone."""

GRAFFITI_MAX_CHARS = 800
"""A conversation on a wall is kept together, but only up to this length."""

VOICE_MAX_LINES_PER_SECTION = 25
"""How many lines of one situation are kept as a voice sample."""

VOICE_MAX_LINES_PER_CHARACTER = 400
"""Budget for the whole voice profile of one character.

The dumps hold thousands of lines, most of them one-word reactions; the budget
keeps the wordy ones, which are what a manner of speech is actually made of.
"""

VOICE_MIN_LINES_PER_SECTION = 2
"""A single line is not a situation; it is dropped as noise."""

VOICE_MIN_CHARS = 15
VOICE_MIN_WORDS = 3
"""Filters for the shouty one-liners ("Ugh!", "Watch out!") that carry no style."""

STAGE_DIRECTION_PATTERN = re.compile(r"^[\[(].*[\])]$")
"""A situation note such as "[When the other survivors are in the safe room]"."""

VERSION_NOTE_PATTERN = re.compile(r"\s*\([^)]*\bversions?\b[^)]*\)\s*$", re.IGNORECASE)
"""Wiki bookkeeping such as "(2 versions)" or "(2 versions, and on Intro)"."""

GAME_SECTION_TITLES = frozenset({"Left 4 Dead", "Left 4 Dead 2"})
"""Top level of the graffiti page; the game name adds nothing to a location."""

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
QUOTE_WRAPPER_PATTERN = re.compile(r'^["“”\'«»\s]+|["“”\'«»\s]+$')
EMPTY_NOISE_PATTERN = re.compile(r"^[.\s…·,;:!?()-]+$")


class CorpusParser(Protocol):
    """Turns the markup of one page into corpus records."""

    def parse(
        self,
        source: WikiSource,
        wikitext: str,
        *,
        source_url: str,
    ) -> list[LoreChunkCreate]:
        """Return the records extracted from ``wikitext``."""
        ...


def split_into_chunks(text: str, *, max_chars: int) -> list[str]:
    """Split ``text`` on paragraph boundaries, then on sentences, under ``max_chars``.

    Paragraph boundaries are preferred because they keep a fact together with
    its explanation; sentences are only used to break up a single paragraph that
    is still too long.
    """
    chunks: list[str] = []
    current = ""
    for paragraph in (block.strip() for block in text.split("\n\n")):
        if not paragraph:
            continue
        if not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return [piece for chunk in chunks for piece in _break_long_chunk(chunk, max_chars)]


def _break_long_chunk(chunk: str, max_chars: int) -> list[str]:
    """Break a chunk that exceeds ``max_chars`` at sentence, then word, boundaries."""
    if len(chunk) <= max_chars:
        return [chunk]

    pieces: list[str] = []
    current = ""
    for sentence in SENTENCE_SPLIT_PATTERN.split(chunk):
        if len(sentence) > max_chars:
            pieces.extend(_break_long_chunk_by_words(sentence, max_chars))
            current = ""
            continue
        if not current:
            current = sentence
        elif len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}"
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def _break_long_chunk_by_words(sentence: str, max_chars: int) -> list[str]:
    """Last resort: split a single over-long sentence into word windows."""
    pieces: list[str] = []
    current = ""
    for word in sentence.split():
        if not current:
            current = word
        elif len(current) + len(word) + 1 <= max_chars:
            current = f"{current} {word}"
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


class GraffitiParser(CorpusParser):
    """Reads the wall inscriptions of the safe rooms.

    The page groups the graffiti by campaign and chapter, and indents replies
    under the inscription they answer, so each root item plus its indented
    replies becomes one record: the answering chain is the whole point of those
    walls.
    """

    def __init__(self, cleaner: WikitextCleaner) -> None:
        self._cleaner = cleaner

    def parse(
        self,
        source: WikiSource,
        wikitext: str,
        *,
        source_url: str,
    ) -> list[LoreChunkCreate]:
        records: list[LoreChunkCreate] = []
        for section in self._cleaner.sections(wikitext):
            records.extend(self._records_for_section(source, section, source_url))
        return records

    def _records_for_section(
        self,
        source: WikiSource,
        section: Section,
        source_url: str,
    ) -> list[LoreChunkCreate]:
        location = _graffiti_location(section)
        items = self._cleaner.list_items(section.body)
        preamble = self._preamble(section, items)

        texts = ([preamble] if preamble else []) + _threads(items)
        return [
            LoreChunkCreate(
                content=piece,
                source_type=source.parser.source_type,
                topic=source.topic,
                title=location[:256],
                source_url=source_url,
            )
            for text in texts
            for piece in _pack_lines(text.splitlines(), max_chars=GRAFFITI_MAX_CHARS)
        ]

    def _preamble(self, section: Section, items: Sequence[ListItem]) -> str:
        """Return the prose that introduces the inscriptions of a chapter.

        The page occasionally notes where the writings actually are ("these
        writings can be found in a normal room, not a safe room"), which is
        worth keeping as its own record.
        """
        body = self._cleaner.clean(section.body)
        marker_line = body.find("*")
        intro = body[:marker_line] if marker_line != -1 else ""
        cleaned = " ".join(intro.split())
        return cleaned if len(cleaned) >= VOICE_MIN_CHARS else ""


def _threads(items: Sequence[ListItem]) -> list[str]:
    """Group list items into threads, indenting every reply under its root."""
    threads: list[list[str]] = []
    for item in items:
        if item.depth <= 1 or not threads:
            threads.append([item.text])
            continue
        threads[-1].append(f"↳ {item.text}")
    return ["\n".join(thread) for thread in threads]


def _graffiti_location(section: Section) -> str:
    """Return ``Campaign › Chapter`` for a graffiti section.

    The page nests every section under the game it belongs to, and a player
    reading the terminal already knows which game they are playing, so the
    outermost level is dropped from the provenance.
    """
    path = section.path if section.path[0] not in GAME_SECTION_TITLES else section.path[1:]
    return " › ".join(path) or section.title


def _pack_lines(lines: Sequence[str], *, max_chars: int) -> list[str]:
    """Group lines into pieces that stay under ``max_chars``.

    Wall conversations can run long; splitting them on line boundaries keeps
    every inscription whole, which truncating mid-sentence would not.
    """
    pieces: list[str] = []
    current: list[str] = []
    length = 0
    for line in lines:
        if current and length + len(line) + 1 > max_chars:
            pieces.append("\n".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        pieces.append("\n".join(current))
    return pieces


class ArticleParser(CorpusParser):
    """Reads an encyclopaedia page, split by heading and paragraph."""

    def __init__(self, cleaner: WikitextCleaner) -> None:
        self._cleaner = cleaner

    def parse(
        self,
        source: WikiSource,
        wikitext: str,
        *,
        source_url: str,
    ) -> list[LoreChunkCreate]:
        lead, sections = self._sections(wikitext)
        # The lead is the article's own summary; it is kept whatever its size,
        # while a heading must earn its keep with a minimum of text.
        records = self._records_for(lead, source, source_url, require_min_length=False)
        for section in sections:
            records.extend(self._records_for(section, source, source_url, require_min_length=True))
        return records

    def _sections(self, wikitext: str) -> tuple[Section, list[Section]]:
        """Return the lead of the page and its headed sections."""
        lead = self._cleaner.lead(wikitext)
        overview = Section(
            level=1,
            title="Overview",
            body=lead,
            path=("Overview",),
        )
        return overview, self._cleaner.sections(wikitext)

    def _records_for(
        self,
        section: Section,
        source: WikiSource,
        source_url: str,
        *,
        require_min_length: bool,
    ) -> list[LoreChunkCreate]:
        text = self._cleaner.clean(section.body)
        if not text or (require_min_length and len(text) < ARTICLE_MIN_CHARS):
            return []
        return [
            LoreChunkCreate(
                content=chunk,
                source_type=source.parser.source_type,
                topic=source.topic,
                title=section.location(source.page)[:256],
                character_name=source.character_name,
                source_url=source_url,
            )
            for chunk in split_into_chunks(text, max_chars=ARTICLE_MAX_CHARS)
        ]


class QuotesParser(CorpusParser):
    """Reads the dialogue dump of one character into voice samples.

    A dump holds thousands of lines, most of them one-word reactions that say
    nothing about a personality. The parser keeps the lines that carry a voice —
    long enough to show sentence rhythm, wording and pet subjects — groups them
    by the situation the game recorded them for, and bounds how much of each
    character is stored.
    """

    def __init__(self, cleaner: WikitextCleaner) -> None:
        self._cleaner = cleaner

    def parse(
        self,
        source: WikiSource,
        wikitext: str,
        *,
        source_url: str,
    ) -> list[LoreChunkCreate]:
        if source.character_name is None:
            return []

        records: list[LoreChunkCreate] = []
        budget = VOICE_MAX_LINES_PER_CHARACTER
        for section in self._cleaner.sections(wikitext):
            if budget <= 0:
                break
            lines = self._lines(section)
            if len(lines) < VOICE_MIN_LINES_PER_SECTION:
                continue
            lines = lines[: min(VOICE_MAX_LINES_PER_SECTION, budget)]
            budget -= len(lines)
            records.append(
                LoreChunkCreate(
                    content="\n".join(lines),
                    source_type=source.parser.source_type,
                    topic=source.topic,
                    title=f"{source.character_name} · {section.title}"[:256],
                    character_name=source.character_name,
                    source_url=source_url,
                )
            )
        return records

    def _lines(self, section: Section) -> list[str]:
        """Return the usable dialogue lines of one situation, in page order."""
        candidates = [item.text for item in self._cleaner.list_items(section.body)]
        return [line for line in map(self._tidy_line, candidates) if self._is_voice(line)]

    @staticmethod
    def _tidy_line(text: str) -> str:
        """Drop the quoting and the wiki bookkeeping around a recorded line."""
        without_version_note = VERSION_NOTE_PATTERN.sub("", text)
        return QUOTE_WRAPPER_PATTERN.sub("", without_version_note).strip()

    @staticmethod
    def _is_voice(line: str) -> bool:
        """Whether a line is long and wordy enough to show how the character talks."""
        if STAGE_DIRECTION_PATTERN.match(line):
            return False
        if len(line) < VOICE_MIN_CHARS or EMPTY_NOISE_PATTERN.match(line):
            return False
        return len(line.split()) >= VOICE_MIN_WORDS
