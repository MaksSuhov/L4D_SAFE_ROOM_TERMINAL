"""Parsers turning one wiki page into corpus records.

Each parser owns the layout of one :class:`~app.ingestion.sources.ParserKind`
and produces :class:`~app.schemas.LoreChunkCreate` records — flat, self-contained
text with enough provenance to be shown to a player. Splitting is deliberate:

* graffiti becomes one record per place, because the walls hold conversations and a reply is
  meaningless without the inscription it answers;
* articles become one record per paragraph group of a section, so a retrieved
  chunk is small enough to fit several of them into the prompt;
* quotes and subtitles become one record per situation of one character, which is
  the unit a voice sample is useful in.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
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

LEAD_SECTION_TITLE = "Обзор"
"""Name given to the unheaded introduction of an article.

The corpus is read from the Russian wiki, so the synthetic section title is
Russian too — it ends up in the provenance shown next to a source.
"""

STAGE_DIRECTION_PATTERN = re.compile(r"^[\[(].*[\])]$")
"""A situation note such as "[When the other survivors are in the safe room]"."""

VERSION_NOTE_PATTERN = re.compile(r"\s*\([^)]*\bversions?\b[^)]*\)\s*$", re.IGNORECASE)
"""Wiki bookkeeping such as "(2 versions)" or "(2 versions, and on Intro)"."""

TABLE_ROW_PATTERN = re.compile(r"\n[ \t]*\|-")
"""Rows of a wiki table; each ``|-`` opens a new one.

The Russian wiki indents its table furniture, so the leading whitespace is part
of the pattern rather than something the caller strips.
"""

TABLE_CELL_PATTERN = re.compile(r"\n[ \t]*\|(?![}-])")
"""Cells inside a row; ``|-`` and ``|}`` are furniture, not cells."""

SUBTITLE_SPAN_PATTERN = re.compile(r"<span\b[^>]*>(.*?)</span>", re.DOTALL | re.IGNORECASE)
"""The ``<span style="color:…">`` wrapper the wiki puts around every subtitle."""

SUBTITLE_GROUP_PATTERN = re.compile(r'table-group-name"?[^\n]*\|\s*(?P<label>[^\n|]+)')
"""The internal cue name grouping the lines of one situation."""

SUBTITLE_ENGLISH_MARKER = "[english]"
"""Marks the English half of a bilingual subtitle row, which is dropped."""

SPEAKER_PREFIX_PATTERN = re.compile(r"^[^\s:][^:]{0,30}:\s+")
"""The ``Эллис: `` prefix naming the speaker, redundant on a per-character page."""

INSCRIPTION_PLACE_PREFIX = "! colspan"
"""Opens the row naming the place the inscriptions below were found in."""

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

    The Russian wiki lays every chapter out as a two-column table — the original
    English inscription the developers painted, and its Russian translation, one
    place per ``colspan`` header — so a record is one place's walls, titled
    ``Campaign › Place`` and holding the translations. Only the Russian column is
    kept: those walls are read aloud to a player who speaks Russian.

    The bulleted layout of the English wiki is still handled, because the prose
    sections of the same page (the trivia notes) use it, and because a page that
    keeps the older layout must not silently vanish from the corpus.
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
        location = _strip_guillemets(_graffiti_location(section))
        groups = self._inscription_groups(section.body)
        if groups:
            return [
                LoreChunkCreate(
                    content=piece,
                    source_type=source.parser.source_type,
                    topic=source.topic,
                    title=f"{location} › {place}"[:256] if place else location[:256],
                    source_url=source_url,
                )
                for place, lines in groups
                for piece in _pack_lines(lines, max_chars=GRAFFITI_MAX_CHARS)
            ]
        return self._records_from_bullets(source, section, location, source_url)

    def _records_from_bullets(
        self,
        source: WikiSource,
        section: Section,
        location: str,
        source_url: str,
    ) -> list[LoreChunkCreate]:
        """Fall back to the bulleted layout of the English wiki."""
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

    def _inscription_groups(self, body: str) -> list[tuple[str, list[str]]]:
        """Split a chapter table into ``(place, translated inscriptions)`` blocks.

        A row is two cells — the original and the translation — and only the
        second is kept, so a row that lost its translation is dropped rather
        than answered in English.
        """
        groups: list[tuple[str, list[str]]] = []
        place = ""
        lines: list[str] = []

        def flush() -> None:
            nonlocal lines
            if lines:
                groups.append((place, lines))
                lines = []

        for row in TABLE_ROW_PATTERN.split(body):
            marked = [line.strip() for line in row.split("\n") if line.strip()]
            if not marked:
                continue
            if marked[0].startswith(INSCRIPTION_PLACE_PREFIX):
                flush()
                place = _place_name(self._cleaner, marked[0])
                continue
            if marked[0].startswith("!"):
                continue
            cells = [cell.strip() for cell in TABLE_CELL_PATTERN.split(row)]
            cells = [cell for cell in cells if cell]
            if len(cells) < 2:
                continue
            # An inscription can hold inline markup (``<u>ENTIRE</u> CITY``) and
            # explicit line breaks, so the translation is cleaned and split back
            # into one line per inscription before it is packed.
            cleaned = self._cleaner.clean(cells[-1])
            lines.extend(line.strip() for line in cleaned.splitlines() if line.strip())
        flush()
        return groups

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


def _place_name(cleaner: WikitextCleaner, header_line: str) -> str:
    """Return the place named by a ``! colspan`` table header.

    The header reads ``! colspan="2" | <center>«[[Апартаменты]]»</center>``, so
    the label is whatever follows the pipe once the markup is cleaned.
    """
    _, _, label = header_line.partition("|")
    return _strip_guillemets(cleaner.clean(label))


def _strip_guillemets(text: str) -> str:
    """Trim the « » the Russian wiki wraps campaign and place names in."""
    return text.strip().strip("«»").strip()


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
            title=LEAD_SECTION_TITLE,
            body=lead,
            path=(LEAD_SECTION_TITLE,),
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
        return _is_voice_line(line)


class SubtitlesParser(CorpusParser):
    """Reads a bilingual subtitle table into voice samples.

    The Russian wiki stores a character's dialogue as one table per game, and
    every recorded line appears twice: the Russian column first, the English
    original second, both wrapped in a coloured ``<span>``. Only the Russian
    half is kept — it is the language the terminal answers in — and the cue name
    of each block doubles as the situation the lines were recorded for.

    This is a separate layout from :class:`QuotesParser` because the Russian
    wiki has no curated quote pages: its dumps are raw subtitle tables, where
    the English wiki kept tidy lists.

    Unlike :class:`QuotesParser` this parser applies no budget to the whole
    character. That budget existed because a raw dump is mostly one-word barks;
    here the word-count filter and the deduplication have already removed them,
    and capping the result would only cut the character's later lines — the
    finale and campaign-specific dialogue — for no gain.
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
        for label, raw_lines in self._groups(wikitext):
            lines = _unique(
                line for line in map(self._tidy_line, raw_lines) if _is_voice_line(line)
            )
            if len(lines) < VOICE_MIN_LINES_PER_SECTION:
                continue
            lines = lines[:VOICE_MAX_LINES_PER_SECTION]
            title = f"{source.character_name} · {label}" if label else source.character_name
            records.append(
                LoreChunkCreate(
                    content="\n".join(lines),
                    source_type=source.parser.source_type,
                    topic=source.topic,
                    title=title[:256],
                    character_name=source.character_name,
                    source_url=source_url,
                )
            )
        return records

    def _groups(self, wikitext: str) -> list[tuple[str, list[str]]]:
        """Split the table into ``(cue name, recorded lines)`` blocks.

        The table interleaves three things: cue names marking a situation, the
        Russian cell of every line, and the English cell of the same line. The
        English cells are recognised by their ``[english]`` marker and dropped,
        and the cue names become the block boundaries.
        """
        groups: list[tuple[str, list[str]]] = []
        label = ""
        lines: list[str] = []
        for row in TABLE_ROW_PATTERN.split(wikitext):
            header = SUBTITLE_GROUP_PATTERN.search(row)
            if header is not None:
                if lines:
                    groups.append((label, lines))
                    lines = []
                label = self._cleaner.clean(header.group("label")).strip()
                continue
            for cell in TABLE_CELL_PATTERN.split(row):
                if SUBTITLE_ENGLISH_MARKER in cell:
                    continue
                lines.extend(span.group(1) for span in SUBTITLE_SPAN_PATTERN.finditer(cell))
        if lines:
            groups.append((label, lines))
        return groups

    def _tidy_line(self, span: str) -> str:
        """Drop the markup, the speaker and the quoting around one subtitle."""
        cleaned = self._cleaner.clean(span)
        without_speaker = SPEAKER_PREFIX_PATTERN.sub("", cleaned)
        return QUOTE_WRAPPER_PATTERN.sub("", without_speaker).strip()


def _is_voice_line(line: str) -> bool:
    """Whether a line is long and wordy enough to show how the character talks.

    A dump is mostly one-word reactions — "Клоуны!" says nothing about a voice —
    so the threshold is on words, not on the line simply being non-empty.
    """
    if STAGE_DIRECTION_PATTERN.match(line):
        return False
    if len(line) < VOICE_MIN_CHARS or EMPTY_NOISE_PATTERN.match(line):
        return False
    return len(line.split()) >= VOICE_MIN_WORDS


def _unique(lines: Iterable[str]) -> list[str]:
    """Drop repeats, keeping the first occurrence.

    A cue name usually holds several takes of the same line, and the same text
    three times in a row teaches the model nothing about the voice.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return unique
