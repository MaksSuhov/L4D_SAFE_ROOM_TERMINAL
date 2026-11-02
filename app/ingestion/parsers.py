"""Парсеры, превращающие одну страницу вики в записи корпуса.

Каждый парсер отвечает за структуру одного
:class:`~app.ingestion.sources.ParserKind` и порождает записи
:class:`~app.schemas.LoreChunkCreate` — плоский самодостаточный текст с
достаточными сведениями о происхождении, чтобы показать его игроку. Разбиение
намеренное:

* граффити становятся одной записью на место, потому что стены ведут беседу,
  а ответ бессмыслен без надписи, на которую отвечает;
* статьи становятся одной записью на группу абзацев раздела, чтобы найденный
  фрагмент был мал и несколько таких укладывались в промпт;
* цитаты и субтитры становятся одной записью на ситуацию одного персонажа —
  именно в таком объёме голосовой образец полезен.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Protocol

from app.ingestion.sources import WikiSource
from app.ingestion.wikitext import ListItem, Section, WikitextCleaner
from app.schemas import LoreChunkCreate

ARTICLE_MAX_CHARS = 900
"""Верхняя граница официальной записи; примерно один плотный абзац."""

ARTICLE_MIN_CHARS = 120
"""Более короткие разделы присоединяются к соседней записи, а не стоят отдельно."""

GRAFFITI_MAX_CHARS = 800
"""Беседа на стене хранится целиком, но только до этой длины."""

VOICE_MAX_LINES_PER_SECTION = 25
"""Сколько строк одной ситуации сохраняется как голосовой образец."""

VOICE_MAX_LINES_PER_CHARACTER = 400
"""Лимит на весь голосовой профиль одного персонажа.

В дампах тысячи строк, большинство — однословные реакции; лимит сохраняет
многословные, из которых и складывается манера речи.
"""

VOICE_MIN_LINES_PER_SECTION = 2
"""Одна строка — это не ситуация; она отбрасывается как шум."""

VOICE_MIN_CHARS = 15
VOICE_MIN_WORDS = 3
"""Фильтры для крикливых однострочников («Ух!», «Осторожно!»), не несущих стиля."""

LEAD_SECTION_TITLE = "Обзор"
"""Название, даваемое введению статьи без заголовка.

Корпус читается из русской вики, поэтому синтетический заголовок раздела тоже
русский — он попадает в происхождение, показываемое рядом с источником.
"""

STAGE_DIRECTION_PATTERN = re.compile(r"^[\[(].*[\])]$")
"""Пометка о ситуации, например «[Когда остальные выжившие в убежище]»."""

VERSION_NOTE_PATTERN = re.compile(r"\s*\([^)]*\bversions?\b[^)]*\)\s*$", re.IGNORECASE)
"""Вики-служебные пометки вроде «(2 versions)» или «(2 versions, and on Intro)»."""

TABLE_ROW_PATTERN = re.compile(r"\n[ \t]*\|-")
"""Строки вики-таблицы; каждая ``|-`` открывает новую.

Русская вики делает отступ у разметки таблиц, поэтому ведущие пробелы — часть
шаблона, а не то, что срезает вызывающий код.
"""

TABLE_CELL_PATTERN = re.compile(r"\n[ \t]*\|(?![}-])")
"""Ячейки внутри строки; ``|-`` и ``|}`` — служебная разметка, а не ячейки."""

SUBTITLE_SPAN_PATTERN = re.compile(r"<span\b[^>]*>(.*?)</span>", re.DOTALL | re.IGNORECASE)
"""Обёртка ``<span style="color:…">``, которой вики окружает каждый субтитр."""

SUBTITLE_GROUP_PATTERN = re.compile(r'table-group-name"?[^\n]*\|\s*(?P<label>[^\n|]+)')
"""Внутреннее имя метки, группирующее строки одной ситуации."""

SUBTITLE_ENGLISH_MARKER = "[english]"
"""Помечает английскую половину двуязычной строки субтитров, которая отбрасывается."""

SPEAKER_PREFIX_PATTERN = re.compile(r"^[^\s:][^:]{0,30}:\s+")
"""Префикс ``Эллис: ``, называющий говорящего; избыточен на странице одного персонажа."""

INSCRIPTION_PLACE_PREFIX = "! colspan"
"""Открывает строку с названием места, где найдены надписи ниже."""

GAME_SECTION_TITLES = frozenset({"Left 4 Dead", "Left 4 Dead 2"})
"""Верхний уровень страницы граффити; название игры ничего не добавляет к локации."""

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
QUOTE_WRAPPER_PATTERN = re.compile(r'^["“”\'«»\s]+|["“”\'«»\s]+$')
EMPTY_NOISE_PATTERN = re.compile(r"^[.\s…·,;:!?()-]+$")


class CorpusParser(Protocol):
    """Превращает разметку одной страницы в записи корпуса."""

    def parse(
        self,
        source: WikiSource,
        wikitext: str,
        *,
        source_url: str,
    ) -> list[LoreChunkCreate]:
        """Вернуть записи, извлечённые из ``wikitext``."""
        ...


def split_into_chunks(text: str, *, max_chars: int) -> list[str]:
    """Разбить ``text`` по границам абзацев, затем по предложениям, в пределах ``max_chars``.

    Границы абзацев предпочтительнее: они сохраняют факт вместе с его пояснением;
    предложения используются лишь чтобы разбить один абзац, который всё ещё
    слишком длинный.
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
    """Разбить фрагмент, превышающий ``max_chars``, по предложениям, затем по словам."""
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
    """Крайний случай: разбить одно слишком длинное предложение на окна по словам."""
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
    """Читает настенные надписи убежищ.

    Русская вики раскладывает каждую главу в таблицу из двух колонок — оригинальная
    английская надпись, которую нарисовали разработчики, и её русский перевод, по
    одному месту на заголовок ``colspan`` — поэтому запись — это стены одного
    места с заголовком ``Campaign › Place``, содержащие переводы. Сохраняется
    только русская колонка: эти стены читаются вслух игроку, говорящему по-русски.

    Разметка списком из английской вики тоже обрабатывается, потому что её
    используют прозаические разделы той же страницы (заметки), и потому что
    страница со старым оформлением не должна молча исчезнуть из корпуса.
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
        """Запасной путь — разметка списком из английской вики."""
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
        """Разбить таблицу главы на блоки ``(place, translated inscriptions)``.

        Строка — это две ячейки, оригинал и перевод, и сохраняется только вторая,
        поэтому строка, потерявшая перевод, отбрасывается, а не отвечается
        на английском.
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
        """Вернуть прозу, вводящую надписи главы.

        Страница иногда отмечает, где на самом деле находятся надписи («эти
        надписи можно найти в обычной комнате, а не в убежище»), а это стоит
        сохранить отдельной записью.
        """
        body = self._cleaner.clean(section.body)
        marker_line = body.find("*")
        intro = body[:marker_line] if marker_line != -1 else ""
        cleaned = " ".join(intro.split())
        return cleaned if len(cleaned) >= VOICE_MIN_CHARS else ""


def _threads(items: Sequence[ListItem]) -> list[str]:
    """Сгруппировать элементы списка в ветки, с отступом каждого ответа под его корнем."""
    threads: list[list[str]] = []
    for item in items:
        if item.depth <= 1 or not threads:
            threads.append([item.text])
            continue
        threads[-1].append(f"↳ {item.text}")
    return ["\n".join(thread) for thread in threads]


def _graffiti_location(section: Section) -> str:
    """Вернуть ``Campaign › Chapter`` для раздела граффити.

    Страница вкладывает каждый раздел в игру, к которой он относится, а игрок,
    читающий терминал, уже знает, в какую игру играет, поэтому самый внешний
    уровень отбрасывается из происхождения.
    """
    path = section.path if section.path[0] not in GAME_SECTION_TITLES else section.path[1:]
    return " › ".join(path) or section.title


def _place_name(cleaner: WikitextCleaner, header_line: str) -> str:
    """Вернуть место, названное заголовком таблицы ``! colspan``.

    Заголовок выглядит как ``! colspan="2" | <center>«[[Апартаменты]]»</center>``,
    поэтому подписью служит всё, что идёт после вертикальной черты, когда
    разметка очищена.
    """
    _, _, label = header_line.partition("|")
    return _strip_guillemets(cleaner.clean(label))


def _strip_guillemets(text: str) -> str:
    """Срезать « », в которые русская вики оборачивает названия кампаний и мест."""
    return text.strip().strip("«»").strip()


def _pack_lines(lines: Sequence[str], *, max_chars: int) -> list[str]:
    """Сгруппировать строки в куски, укладывающиеся в ``max_chars``.

    Настенные беседы бывают длинными; разбиение по границам строк сохраняет
    каждую надпись целой, чего обрезание на середине предложения не даёт.
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
    """Читает энциклопедическую страницу, разбитую по заголовкам и абзацам."""

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
        """Вернуть введение страницы и её разделы с заголовками."""
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
    """Читает дамп диалогов одного персонажа в голосовые образцы.

    В дампе тысячи строк, большинство — однословные реакции, ничего не говорящие
    о личности. Парсер сохраняет строки, несущие голос, — достаточно длинные,
    чтобы показать ритм фраз, формулировки и любимые темы, — группирует их по
    ситуации, для которой игра их записала, и ограничивает, сколько хранится от
    каждого персонажа.
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
        """Вернуть пригодные строки диалога одной ситуации в порядке страницы."""
        candidates = [item.text for item in self._cleaner.list_items(section.body)]
        return [line for line in map(self._tidy_line, candidates) if self._is_voice(line)]

    @staticmethod
    def _tidy_line(text: str) -> str:
        """Убрать кавычки и вики-служебные пометки вокруг записанной строки."""
        without_version_note = VERSION_NOTE_PATTERN.sub("", text)
        return QUOTE_WRAPPER_PATTERN.sub("", without_version_note).strip()

    @staticmethod
    def _is_voice(line: str) -> bool:
        """Достаточно ли строка длинна и многословна, чтобы показать манеру речи персонажа."""
        return _is_voice_line(line)


class SubtitlesParser(CorpusParser):
    """Читает двуязычную таблицу субтитров в голосовые образцы.

    Русская вики хранит диалоги персонажа как одну таблицу на игру, и каждая
    записанная строка появляется дважды: сначала русская колонка, потом английский
    оригинал, обе обёрнуты в цветной ``<span>``. Сохраняется только русская
    половина — это язык, на котором отвечает терминал, — а имя метки каждого блока
    служит ситуацией, для которой строки были записаны.

    Это отдельная разметка от :class:`QuotesParser`, потому что на русской вики
    нет отобранных страниц цитат: её дампы — сырые таблицы субтитров, тогда как
    английская вики держала аккуратные списки.

    В отличие от :class:`QuotesParser` этот парсер не применяет лимит ко всему
    персонажу. Тот лимит существовал, потому что сырой дамп — это в основном
    однословные выкрики; здесь фильтр по словам и дедупликация уже убрали их, а
    ограничение результата лишь отрезало бы более поздние строки персонажа —
    финал и диалоги конкретных кампаний — без всякой пользы.
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
        """Разбить таблицу на блоки ``(cue name, recorded lines)``.

        Таблица чередует три вещи: имена меток, отмечающие ситуацию, русскую ячейку
        каждой строки и английскую ячейку той же строки. Английские ячейки узнаются
        по маркеру ``[english]`` и отбрасываются, а имена меток становятся границами
        блоков.
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
        """Убрать разметку, говорящего и кавычки вокруг одного субтитра."""
        cleaned = self._cleaner.clean(span)
        without_speaker = SPEAKER_PREFIX_PATTERN.sub("", cleaned)
        return QUOTE_WRAPPER_PATTERN.sub("", without_speaker).strip()


def _is_voice_line(line: str) -> bool:
    """Достаточно ли строка длинна и многословна, чтобы показать манеру речи персонажа.

    Дамп — это в основном однословные реакции — «Клоуны!» ничего не говорит о
    голосе, — поэтому порог по словам, а не по тому, что строка просто непустая.
    """
    if STAGE_DIRECTION_PATTERN.match(line):
        return False
    if len(line) < VOICE_MIN_CHARS or EMPTY_NOISE_PATTERN.match(line):
        return False
    return len(line.split()) >= VOICE_MIN_WORDS


def _unique(lines: Iterable[str]) -> list[str]:
    """Убрать повторы, сохраняя первое вхождение.

    Имя метки обычно содержит несколько дублей одной строки, а один и тот же
    текст три раза подряд ничему не учит модель о голосе.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return unique
