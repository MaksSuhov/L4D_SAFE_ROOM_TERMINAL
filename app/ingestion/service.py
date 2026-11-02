"""Сборка корпуса: загрузка, разбор, нормализация, дедупликация, снимок.

Это композиционный корень этапа загрузки. Он знает порядок шагов и ничего об
их внутренностях: клиент вики забирает, парсеры извлекают, переводчик
переписывает, а снимок сохраняет. Замена любого из них — другой источник,
другая политика нарезки на фрагменты, полный отказ от перевода — это вопрос
подстановки другого партнёра.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from app.ingestion.parsers import CorpusParser
from app.ingestion.snapshot import CorpusSnapshot
from app.ingestion.sources import WIKI_SOURCES, ParserKind, WikiSource
from app.ingestion.translation import CorpusTranslator
from app.ingestion.wiki_client import MediaWikiClient
from app.schemas import LoreChunkCreate

logger = logging.getLogger("l4d.ingestion")


@dataclass(frozen=True, slots=True)
class CorpusBuildReport:
    """Что дала одна сборка корпуса, для сводки CLI и тестов."""

    pages: int = 0
    documents: int = 0
    duplicates: int = 0
    translated: int = 0
    by_source_type: dict[str, int] = field(default_factory=dict)
    by_topic: dict[str, int] = field(default_factory=dict)


class WikiCorpusBuilder:
    """Превращает объявленные источники вики в нормализованный корпус."""

    def __init__(
        self,
        client: MediaWikiClient,
        parsers: Mapping[ParserKind, CorpusParser],
        snapshot: CorpusSnapshot,
        *,
        translator: CorpusTranslator | None = None,
        sources: Sequence[WikiSource] = WIKI_SOURCES,
    ) -> None:
        self._client = client
        self._parsers = parsers
        self._snapshot = snapshot
        self._translator = translator
        self._sources = sources

    async def build(self, *, refresh: bool = False, translate: bool = False) -> CorpusBuildReport:
        """Забрать и разобрать каждый источник, затем сохранить снимок корпуса.

        ``refresh`` перечитывает вики вместо локального кэша разметки, а
        ``translate`` сначала приводит результат к языку корпуса.
        """
        pages = await self._fetch(reindex_markup=refresh)
        documents, duplicates = self._parse(pages)

        translated = 0
        if translate and self._translator is not None:
            outcome = await self._translator.translate(documents)
            documents = outcome.documents
            translated = outcome.translated

        self._snapshot.write(documents)
        report = CorpusBuildReport(
            pages=len(pages),
            documents=len(documents),
            duplicates=duplicates,
            translated=translated,
            by_source_type=_tally(str(document.source_type) for document in documents),
            by_topic=_tally(str(document.topic) for document in documents),
        )
        logger.info(
            "Corpus built from %d page(s): %d record(s), %d duplicate(s) dropped, %d translated",
            report.pages,
            report.documents,
            report.duplicates,
            report.translated,
        )
        logger.info("Records per tier: %s", report.by_source_type)
        logger.info("Records per topic: %s", report.by_topic)
        return report

    async def _fetch(self, *, reindex_markup: bool) -> dict[str, str]:
        """Забрать каждую объявленную страницу, сообщая о ходе работы."""
        titles = [source.page for source in self._sources]
        logger.info(
            "Fetching %d wiki page(s)%s",
            len(titles),
            " (refresh)" if reindex_markup else "",
        )
        pages: dict[str, str] = {}
        for position, title in enumerate(titles, start=1):
            fetched = await self._client.fetch_pages([title], refresh=reindex_markup)
            if fetched:
                pages.update(fetched)
            logger.info("[%d/%d] %s", position, len(titles), title)
        return pages

    def _parse(self, pages: Mapping[str, str]) -> tuple[list[LoreChunkCreate], int]:
        """Прогнать каждую страницу через её парсер и отбросить дубликаты.

        Страницу можно получить дважды — через перенаправление или потому что два
        раздела содержат одну и ту же надпись, — а один и тот же текст, проиндексированный
        дважды, сместит поиск в пользу того, кто его повторил.
        """
        documents: list[LoreChunkCreate] = []
        seen: set[tuple[str, str]] = set()
        duplicates = 0
        for source in self._sources:
            wikitext = pages.get(source.page)
            if wikitext is None:
                logger.warning("No markup for %s, skipping", source.page)
                continue
            parser = self._parsers[source.parser]
            extracted = parser.parse(
                source, wikitext, source_url=self._client.page_url(source.page)
            )
            kept = [document for document in extracted if _remember(seen, document)]
            duplicates += len(extracted) - len(kept)
            logger.info("  %s → %d record(s)", source.page, len(kept))
            documents.extend(kept)
        return documents, duplicates


def _remember(seen: set[tuple[str, str]], document: LoreChunkCreate) -> bool:
    """Отметить запись как увиденную, сообщив, новая ли она."""
    key = (str(document.source_type), document.content)
    if key in seen:
        return False
    seen.add(key)
    return True


def _tally(values: Iterable[str]) -> dict[str, int]:
    """Посчитать вхождения каждого значения в устойчивом порядке."""
    return dict(sorted(Counter(values).items()))
