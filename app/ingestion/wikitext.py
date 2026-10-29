"""Turning MediaWiki markup into readable plain text.

The Fandom pages mix prose with heavy furniture: infobox templates, stat tables,
image galleries, reference footnotes and inline styling. A vector store is only
as good as the text fed to it, so this module strips the furniture while keeping
the facts — notably the stat tables, whose numbers are some of the most useful
data on the infected pages.

Everything here is pure text processing: no network, no database, so the
behaviour is pinned down by fast unit tests.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

HEADING_PATTERN = re.compile(r"^(?P<marks>={1,6})\s*(?P<title>.+?)\s*(?P=marks)\s*$")
"""A wiki heading such as ``=== The Subway ===``."""

LIST_MARKER_PATTERN = re.compile(r"^(?P<marks>[*#:;]+)\s*(?P<body>.*)$")
"""A list item, with the marker chain kept to recover its nesting depth."""

COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
GALLERY_PATTERN = re.compile(r"<gallery\b[^>]*>.*?</gallery>", re.DOTALL)
REF_PATTERN = re.compile(r"<ref\b[^>]*?/>|<ref\b[^>]*?>.*?</ref>", re.DOTALL)
BREAK_PATTERN = re.compile(r"<br\s*/?>", re.IGNORECASE)
TAG_PATTERN = re.compile(r"</?[a-zA-Z][^>]*>")
EMPHASIS_BOLD_ITALIC_PATTERN = re.compile(r"'{5}")
BOLD_PATTERN = re.compile(r"'{3}")
ITALIC_PATTERN = re.compile(r"'{2}")
MAGIC_WORD_PATTERN = re.compile(r"__[A-Z]+__")
ATTRIBUTE_PATTERN = re.compile(
    r"\b(?:style|class|id|width|height|align|valign|colspan|rowspan|title|alt)\s*=\s*"
    r'(?:"[^"]*"|\'[^\']*\'|[^\s|}\]]+)'
)
BLANK_LINE_RUN = re.compile(r"\n{3,}")
INLINE_SPACE_RUN = re.compile(r"[ \t]{2,}")

DROPPED_MEDIA_PREFIXES = ("file:", "image:", "category:", "w:", "wikipedia:")

INFOBOX_PREFIX = "infobox"
"""Templates whose name starts with this are data, not layout, and are kept.

The infection pages store the numbers that make them worth reading — health per
difficulty, damage, speed — in ``{{Infobox infected|Health = 4000 …}}``. Dropping
every template would drop exactly the facts the archive is supposed to answer
questions with, so infoboxes are rendered as ``Key — Value`` lines while layout
templates such as ``{{pix}}`` are still removed.
"""

INFOBOX_PAIR_SEPARATOR = " — "
INFOBOX_IGNORED_KEYS = frozenset({"image"})
"""Keys whose value is a picture, not information."""

EMPTY_PAIR_PATTERN = re.compile(r"^[^—\n]{1,80}—$")
"""A parameter whose value was markup only, e.g. ``Ability icon —``.

Such a row survives rendering but says nothing, and a bag of empty keys is what
makes an embedding diffuse — precisely the opposite of what a statistics record
is for.
"""

CELL_SEPARATOR_PATTERN = re.compile(r"\s*(?:\|\||!!)\s*")


@dataclass(frozen=True, slots=True)
class Section:
    """One heading of a page together with the text that follows it."""

    level: int
    title: str
    body: str
    path: tuple[str, ...]
    """Titles of the enclosing headings, outermost first, ending with ``title``."""

    def location(self, page: str) -> str:
        """Human-readable provenance such as ``Green Flu › Origin``."""
        return " › ".join((page, *self.path))


@dataclass(frozen=True, slots=True)
class ListItem:
    """One list item with the depth of its marker chain."""

    depth: int
    text: str


def _find_balanced_end(text: str, start: int, open_token: str, close_token: str) -> int:
    """Return the index just past the token that closes the one at ``start``.

    Nested occurrences of ``open_token`` are counted, which is what makes the
    scanner survive constructs such as a file link whose caption contains
    another link.
    """
    depth = 0
    index = start
    open_length = len(open_token)
    close_length = len(close_token)
    while index < len(text):
        if text.startswith(open_token, index):
            depth += 1
            index += open_length
        elif text.startswith(close_token, index):
            depth -= 1
            index += close_length
            if depth == 0:
                return index
        else:
            index += 1
    return len(text)


def _split_template_parameters(text: str) -> list[str]:
    """Split template parameters on the ``|`` that separates them.

    A naive split is wrong, and expensively so: a value such as
    ``[[File:x.png|50px]]`` contains a pipe of its own, so splitting on every pipe
    tears the link apart. The leftover ``[[`` is unbalanced, and the link
    resolver then swallows the rest of the value — which is how the Tank's health
    figures disappeared the first time this was written. Only top-level pipes
    separate parameters; those inside links, nested templates or tags do not.
    """
    parts: list[str] = []
    current: list[str] = []
    links = 0
    templates = 0
    index = 0
    while index < len(text):
        if text.startswith("[[", index):
            links += 1
            current.append("[[")
            index += 2
            continue
        if text.startswith("]]", index) and links:
            links -= 1
            current.append("]]")
            index += 2
            continue
        if text.startswith("{{", index):
            templates += 1
            current.append("{{")
            index += 2
            continue
        if text.startswith("}}", index) and templates:
            templates -= 1
            current.append("}}")
            index += 2
            continue
        character = text[index]
        if character == "|" and not links and not templates:
            parts.append("".join(current))
            current = []
        else:
            current.append(character)
        index += 1
    parts.append("".join(current))
    return parts


def _render_infobox(inner: str) -> str:
    """Render the parameters of an infobox template, or return ``""`` to drop it.

    ``inner`` is the text between the braces of one balanced ``{{…}}`` span.
    """
    segments = _split_template_parameters(inner)
    if not segments[0].strip().lower().startswith(INFOBOX_PREFIX):
        return ""

    lines: list[str] = []
    for segment in segments[1:]:
        key, separator, value = segment.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key or key.lower() in INFOBOX_IGNORED_KEYS or not value.strip():
            continue
        lines.append(f"{key}{INFOBOX_PAIR_SEPARATOR}{value.strip()}")
    # Blank lines around the block keep it a paragraph of its own, so the
    # statistics of a page land in their own retrievable record instead of
    # being glued to the article introduction.
    return f"\n\n{chr(10).join(lines)}\n\n" if lines else ""


def _resolve_templates(text: str) -> str:
    """Keep the data of infoboxes and drop every other template."""
    result: list[str] = []
    index = 0
    while index < len(text):
        start = text.find("{{", index)
        if start == -1:
            result.append(text[index:])
            break
        result.append(text[index:start])
        end = _find_balanced_end(text, start, "{{", "}}")
        inner = text[start + 2 : max(end - 2, start + 2)]
        result.append(_render_infobox(inner))
        index = end
    return "".join(result)


def _drop_balanced_spans(text: str, open_token: str, close_token: str) -> str:
    """Remove every balanced ``open_token``…``close_token`` span from ``text``."""
    result: list[str] = []
    index = 0
    while index < len(text):
        start = text.find(open_token, index)
        if start == -1:
            result.append(text[index:])
            break
        result.append(text[index:start])
        index = _find_balanced_end(text, start, open_token, close_token)
    return "".join(result)


def _resolve_links(text: str) -> str:
    """Replace wiki links by their label, dropping links to media and categories."""
    result: list[str] = []
    index = 0
    while index < len(text):
        start = text.find("[[", index)
        if start == -1:
            result.append(text[index:])
            break
        result.append(text[index:start])
        end = _find_balanced_end(text, start, "[[", "]]")
        inner = text[start + 2 : max(end - 2, start + 2)]
        result.append(_link_label(inner))
        index = end
    return "".join(result)


def _link_label(inner: str) -> str:
    """Return the visible text of one link target, or an empty string to drop it."""
    parts = inner.split("|")
    target = parts[0].strip()
    lowered = target.lower()
    if lowered.startswith(DROPPED_MEDIA_PREFIXES):
        return ""
    if len(parts) > 1:
        label = parts[-1].strip()
        return label or target
    return target.split("#", 1)[0].strip()


def _flatten_tables(text: str) -> str:
    """Turn wiki table markup into plain lines, preserving the cell text.

    The stat tables of the infected pages hold the health, damage and speed
    numbers, so a table is flattened rather than deleted: each row becomes one
    line and the cells of a row are joined with a dash.
    """
    lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("{|", "|}")):
            continue
        if stripped.startswith("|-"):
            lines.append("")
            continue
        if stripped.startswith(("!", "|")):
            cells = [
                cell.strip(" |")
                for cell in CELL_SEPARATOR_PATTERN.split(stripped.lstrip("!|").strip())
            ]
            populated = [cell for cell in cells if cell]
            if populated:
                lines.append(" — ".join(populated))
            continue
        lines.append(line)
    return "\n".join(lines)


class WikitextCleaner:
    """Converts MediaWiki markup into plain text with the structure preserved."""

    def clean(self, wikitext: str) -> str:
        """Return the readable text of ``wikitext``, line breaks preserved."""
        # Comments and footnotes are inline, so they vanish without leaving a
        # gap behind: "infection.<ref>a</ref>" must not become "infection. ."
        text = COMMENT_PATTERN.sub("", wikitext)
        text = REF_PATTERN.sub("", text)
        text = GALLERY_PATTERN.sub(" ", text)
        # Infoboxes first, so their data survives; a second sweep then drops the
        # templates nested inside them.
        text = _drop_balanced_spans(_resolve_templates(text), "{{", "}}")
        text = ATTRIBUTE_PATTERN.sub(" ", text)
        text = _flatten_tables(text)
        text = _resolve_links(text)
        text = BREAK_PATTERN.sub("\n", text)
        text = TAG_PATTERN.sub(" ", text)
        text = EMPHASIS_BOLD_ITALIC_PATTERN.sub("", text)
        text = BOLD_PATTERN.sub("", text)
        text = ITALIC_PATTERN.sub("", text)
        text = MAGIC_WORD_PATTERN.sub(" ", text)
        text = html.unescape(text).replace("\xa0", " ")
        return self._tidy(text)

    def lead(self, wikitext: str) -> str:
        """Return the text that precedes the first heading.

        The lead is the page's own summary and carries the most cited facts, so
        it is treated as a section of its own rather than dropped by
        :meth:`sections`, which starts at the first heading.
        """
        collected: list[str] = []
        for line in wikitext.split("\n"):
            if HEADING_PATTERN.match(line.strip()):
                break
            collected.append(line)
        return "\n".join(collected).strip()

    def sections(self, wikitext: str, *, min_level: int = 2) -> list[Section]:
        """Split a page into its headings, keeping the nesting path of each one."""
        sections: list[Section] = []
        stack: list[tuple[int, str]] = []
        current_title: str | None = None
        current_level = 0
        buffer: list[str] = []

        def flush() -> None:
            if current_title is None:
                return
            sections.append(
                Section(
                    level=current_level,
                    title=current_title,
                    body="\n".join(buffer).strip(),
                    path=tuple(title for _, title in stack),
                )
            )

        for line in wikitext.split("\n"):
            match = HEADING_PATTERN.match(line.strip())
            if match is None:
                buffer.append(line)
                continue
            level = len(match.group("marks"))
            if level < min_level:
                buffer.append(line)
                continue
            flush()
            title = self.clean(match.group("title")).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_title = title
            current_level = level
            buffer = []

        flush()
        return sections

    def list_items(self, text: str) -> list[ListItem]:
        """Extract list items with their nesting depth, skipping blank markers.

        A deeper marker is a reply to the item above it — the safe-room walls are
        full of such back-and-forth — so the depth is reported instead of being
        flattened away by the caller.
        """
        items: list[ListItem] = []
        for line in self.clean(text).split("\n"):
            match = LIST_MARKER_PATTERN.match(line.strip())
            if match is None:
                continue
            body = match.group("body").strip()
            if not body:
                continue
            items.append(ListItem(depth=len(match.group("marks")), text=body))
        return items

    def paragraphs(self, text: str) -> list[str]:
        """Split cleaned text into non-empty paragraphs."""
        return [block.strip() for block in re.split(r"\n\s*\n", self.clean(text)) if block.strip()]

    @staticmethod
    def _tidy(text: str) -> str:
        """Normalise spacing without losing the paragraph structure."""
        lines = [
            INLINE_SPACE_RUN.sub(" ", line).strip()
            for line in text.split("\n")
            if not EMPTY_PAIR_PATTERN.match(line.strip())
        ]
        return BLANK_LINE_RUN.sub("\n\n", "\n".join(lines)).strip()
