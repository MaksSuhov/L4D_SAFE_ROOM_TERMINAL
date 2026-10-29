"""Corpus-acquisition stage: the Left 4 Dead Fandom wiki as a lore archive.

The package is an infrastructure adapter. It reads somebody else's wiki, turns
it into flat, self-contained records tagged with how much they can be trusted,
and hands them to the service layer — which embeds and stores them without ever
knowing where they came from.

The pipeline is four steps, each replaceable on its own:

``wiki_client``
    Fetches raw markup, with a disk cache and polite retries.
``wikitext``
    Cleans MediaWiki markup down to readable plain text.
``parsers``
    Splits a page into records according to its layout.
``translation``
    Optional: normalises the corpus into the language the terminal answers in.
"""

from app.ingestion.parsers import ArticleParser, CorpusParser, GraffitiParser, QuotesParser
from app.ingestion.service import CorpusBuildReport, WikiCorpusBuilder
from app.ingestion.snapshot import CorpusSnapshot
from app.ingestion.sources import WIKI_SOURCES, ParserKind, WikiSource
from app.ingestion.translation import CorpusTranslator, TranslationOutcome
from app.ingestion.wiki_client import MediaWikiClient, WikiPageNotFoundError, WikitextCache
from app.ingestion.wikitext import ListItem, Section, WikitextCleaner

__all__ = [
    "WIKI_SOURCES",
    "ArticleParser",
    "CorpusBuildReport",
    "CorpusParser",
    "CorpusSnapshot",
    "CorpusTranslator",
    "GraffitiParser",
    "ListItem",
    "MediaWikiClient",
    "ParserKind",
    "QuotesParser",
    "Section",
    "TranslationOutcome",
    "WikiCorpusBuilder",
    "WikiPageNotFoundError",
    "WikiSource",
    "WikitextCache",
    "WikitextCleaner",
]
