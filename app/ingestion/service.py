"""Building the corpus: fetch, parse, normalise, deduplicate, snapshot.

This is the composition root of the ingestion stage. It knows the order of the
steps and nothing about their internals: the wiki client fetches, the parsers
extract, the translator rewrites and the snapshot stores. Swapping any of them —
a different wiki, a different chunking policy, no translation at all — is a
matter of injecting a different collaborator.
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
    """What one corpus build produced, for the CLI summary and the tests."""

    pages: int = 0
    documents: int = 0
    duplicates: int = 0
    translated: int = 0
    by_source_type: dict[str, int] = field(default_factory=dict)
    by_topic: dict[str, int] = field(default_factory=dict)


class WikiCorpusBuilder:
    """Turns the declared wiki sources into a normalised corpus."""

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
        """Fetch and parse every source, then persist the corpus snapshot.

        ``refresh`` re-reads the wiki instead of the local markup cache and
        ``translate`` normalises the result into the corpus language first.
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
        """Fetch every declared page, reporting progress as it goes."""
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
        """Run every page through its parser and drop the duplicates.

        A page can be reached twice — through a redirect, or because two
        sections hold the same inscription — and the same text embedded twice
        would skew retrieval towards whoever repeated it.
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
    """Register a record as seen, reporting whether it is new."""
    key = (str(document.source_type), document.content)
    if key in seen:
        return False
    seen.add(key)
    return True


def _tally(values: Iterable[str]) -> dict[str, int]:
    """Count the occurrences of every value, in a stable order."""
    return dict(sorted(Counter(values).items()))
