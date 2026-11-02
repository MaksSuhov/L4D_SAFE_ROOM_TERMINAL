"""Превращение разметки MediaWiki в читаемый простой текст.

Страницы Fandom смешивают прозу с тяжёлой служебной разметкой: шаблонами
инфобоксов, таблицами характеристик, галереями изображений, сносками и
встроенным стилем. Векторное хранилище ровно настолько хорошо, насколько хорош
скормленный ему текст, поэтому модуль срезает служебную разметку, сохраняя
факты — прежде всего таблицы характеристик, чьи числа — одни из самых полезных
данных на страницах заражённых.

Здесь всё — чистая обработка текста: ни сети, ни базы, поэтому поведение
закреплено быстрыми модульными тестами.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

HEADING_PATTERN = re.compile(r"^(?P<marks>={1,6})\s*(?P<title>.+?)\s*(?P=marks)\s*$")
"""Вики-заголовок, например ``=== The Subway ===``."""

LIST_MARKER_PATTERN = re.compile(r"^(?P<marks>[*#:;]+)\s*(?P<body>.*)$")
"""Элемент списка; цепочка маркеров сохраняется, чтобы восстановить глубину вложенности."""

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
EMPTY_PARENTHESES_PATTERN = re.compile(r"[(\[]\s*[)\]]")
"""Скобки, оставшиеся пустыми после удалённого шаблона, напр. ``Зелёный грипп ()``.

Шаблон вроде ``{{lang-en|Green Flu}}`` — служебная разметка, но скобки вокруг
него принадлежат предложению, поэтому удаление шаблона оставляет пустую пару,
которая читается как пропавший факт.
"""

STRAY_MARKUP_PATTERN = re.compile(r"\[\[|\]\]|\{\{|\}\}")
"""Сдвоенные скобки, пережившие разрешение ссылок и шаблонов.

Вики написана руками и иногда опечатывает ссылку — ``[[штурмовая
винтовка]]M16]]`` — что оставляет болтающуюся ``]]``, приклеенную к
следующему слову. После разрешения ссылок и шаблонов сдвоенная скобка может
быть только мусором.
"""

INTERWIKI_PREFIX_PATTERN = re.compile(r"^[a-z]{2,3}:")
"""Языковая или интервики-цель ссылки, например ``en:Carriers`` или ``es:La Batalla Final``.

Каждая статья Fandom заканчивается блоком таких ссылок, и лора в них нет: если
оставить, каждая страница даёт запись, состоящую только из имён ссылок, а та
претендует на место в доказательствах, которое никогда не заполнит. Собственные
заголовки статей вики кириллические или с префиксом пространства имён, поэтому
латинский префикс из двух-трёх букв — всегда ссылка, никогда не тема.
"""

DROPPED_MEDIA_PREFIXES = (
    "file:",
    "image:",
    "category:",
    "w:",
    "wikipedia:",
    # The Russian wiki uses localised namespaces, and a link to ``Файл:`` that
    # slips through would put a caption into the corpus as if it were lore.
    "файл:",
    "изображение:",
    "категория:",
)

INFOBOX_PREFIX = "infobox"
"""Шаблоны, чьё имя начинается с этого, — данные, а не разметка, и сохраняются.

Страницы заражения хранят числа, ради которых их и стоит читать — здоровье по
сложности, урон, скорость — в ``{{Infobox infected|Health = 4000 …}}``. Удаление
каждого шаблона удалило бы ровно те факты, которыми архив и должен отвечать на
вопросы, поэтому инфобоксы рендерятся строками ``Key — Value``, а шаблоны
разметки вроде ``{{pix}}`` по-прежнему убираются.
"""

INFOBOX_PAIR_SEPARATOR = " — "
INFOBOX_IGNORED_KEYS = frozenset({"image"})
"""Ключи, значением которых служит картинка, а не информация."""

EMPTY_PAIR_PATTERN = re.compile(r"^[^—\n]{1,80}—$")
"""Параметр, значением которого была одна разметка, напр. ``Ability icon —``.

Такая строка переживает рендеринг, но ничего не говорит, а мешок пустых ключей
делает эмбеддинг размытым — ровно наоборот тому, для чего нужна запись
статистики.
"""

CELL_SEPARATOR_PATTERN = re.compile(r"\s*(?:\|\||!!)\s*")


@dataclass(frozen=True, slots=True)
class Section:
    """Один заголовок страницы вместе с текстом, который за ним следует."""

    level: int
    title: str
    body: str
    path: tuple[str, ...]
    """Заголовки охватывающих разделов, от внешнего к внутреннему, кончая ``title``."""

    def location(self, page: str) -> str:
        """Читаемое происхождение, например ``Green Flu › Origin``."""
        return " › ".join((page, *self.path))


@dataclass(frozen=True, slots=True)
class ListItem:
    """Один элемент списка с глубиной его цепочки маркеров."""

    depth: int
    text: str


def _find_balanced_end(text: str, start: int, open_token: str, close_token: str) -> int:
    """Вернуть индекс сразу за токеном, закрывающим токен в ``start``.

    Вложенные вхождения ``open_token`` учитываются — именно это позволяет
    сканированию переживать конструкции вроде файловой ссылки, чья подпись
    содержит другую ссылку.
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
    """Разбить параметры шаблона по ``|``, который их разделяет.

    Наивное разбиение неверно, и дорого: значение вроде ``[[File:x.png|50px]]``
    содержит свою вертикальную черту, поэтому разбиение по каждой черте разрывает
    ссылку. Оставшийся ``[[`` не сбалансирован, и разрешение ссылок затем
    проглатывает остаток значения — так в первый раз и исчезли цифры здоровья
    Танка. Разделяют параметры только черты верхнего уровня; те, что внутри
    ссылок, вложенных шаблонов или тегов, — нет.
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
    """Отрендерить параметры шаблона инфобокса или вернуть ``""``, чтобы его отбросить.

    ``inner`` — текст между скобками одного сбалансированного пролёта ``{{…}}``.
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
    """Сохранить данные инфобоксов и отбросить все остальные шаблоны."""
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
    """Удалить из ``text`` каждый сбалансированный пролёт ``open_token``…``close_token``."""
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
    """Заменить вики-ссылки их подписью, отбрасывая ссылки на медиа и категории."""
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
    """Вернуть видимый текст одной цели ссылки или пустую строку, чтобы её отбросить.

    Параметры делятся только по *чертам верхнего уровня*: файловая ссылка, чья
    подпись содержит другую ссылку — ``[[Файл:x.png|thumb|Склад [[Боеприпасы|…]]]]``
    — иначе была бы разорвана наивным разбиением, вынося хвост подписи в корпус с
    приклеенной лишней ``]]``.
    """
    parts = _split_template_parameters(inner)
    target = parts[0].strip()
    lowered = target.lower()
    if lowered.startswith(DROPPED_MEDIA_PREFIXES) or INTERWIKI_PREFIX_PATTERN.match(lowered):
        return ""
    if len(parts) > 1:
        label = parts[-1].strip()
        return label or target
    return target.split("#", 1)[0].strip()


def _flatten_tables(text: str) -> str:
    """Превратить разметку вики-таблиц в простые строки, сохраняя текст ячеек.

    Таблицы характеристик страниц заражённых хранят числа здоровья, урона и
    скорости, поэтому таблица разворачивается, а не удаляется: каждая строка
    становится одной строкой текста, а ячейки строки соединяются тире.
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
    """Превращает разметку MediaWiki в простой текст с сохранением структуры."""

    def clean(self, wikitext: str) -> str:
        """Вернуть читаемый текст ``wikitext`` с сохранением переносов строк."""
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
        text = STRAY_MARKUP_PATTERN.sub("", text)
        text = EMPTY_PARENTHESES_PATTERN.sub("", text)
        text = MAGIC_WORD_PATTERN.sub(" ", text)
        text = html.unescape(text).replace("\xa0", " ")
        return self._tidy(text)

    def lead(self, wikitext: str) -> str:
        """Вернуть текст, предшествующий первому заголовку.

        Введение — это собственная сводка страницы, несущая самые цитируемые факты,
        поэтому оно считается отдельным разделом, а не отбрасывается :meth:`sections`,
        который начинается с первого заголовка.
        """
        collected: list[str] = []
        for line in wikitext.split("\n"):
            if HEADING_PATTERN.match(line.strip()):
                break
            collected.append(line)
        return "\n".join(collected).strip()

    def sections(self, wikitext: str, *, min_level: int = 2) -> list[Section]:
        """Разбить страницу на её заголовки, сохраняя путь вложенности каждого."""
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
        """Извлечь элементы списка с их глубиной вложенности, пропуская пустые маркеры.

        Более глубокий маркер — это ответ на элемент выше, а стены убежищ полны
        таких перекличек, поэтому глубина сообщается, а не выравнивается вызывающим кодом.
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
        """Разбить очищенный текст на непустые абзацы."""
        return [block.strip() for block in re.split(r"\n\s*\n", self.clean(text)) if block.strip()]

    @staticmethod
    def _tidy(text: str) -> str:
        """Нормализовать пробелы, не теряя структуру абзацев."""
        lines = [
            INLINE_SPACE_RUN.sub(" ", line).strip()
            for line in text.split("\n")
            if not EMPTY_PAIR_PATTERN.match(line.strip())
        ]
        return BLANK_LINE_RUN.sub("\n\n", "\n".join(lines)).strip()
