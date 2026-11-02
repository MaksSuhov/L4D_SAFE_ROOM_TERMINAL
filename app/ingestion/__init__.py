"""Этап загрузки корпуса: вики Left 4 Dead на Fandom как архив лора.

Пакет — инфраструктурный адаптер. Он читает чужую вики, превращает её в
плоские самодостаточные записи с пометкой о степени доверия и передаёт их
слою сервисов — тот их индексирует и хранит, не зная, откуда они взялись.

Конвейер состоит из четырёх шагов, каждый заменяется отдельно:

``wiki_client``
    Забирает сырую разметку, с дисковым кэшем и вежливыми повторами.
``wikitext``
    Чистит разметку MediaWiki до читаемого простого текста.
``parsers``
    Разбивает страницу на записи по её структуре.
``translation``
    Необязательный: приводит корпус к языку, на котором отвечает терминал.
"""

from app.ingestion.parsers import (
    ArticleParser,
    CorpusParser,
    GraffitiParser,
    QuotesParser,
    SubtitlesParser,
)
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
    "SubtitlesParser",
    "TranslationOutcome",
    "WikiCorpusBuilder",
    "WikiPageNotFoundError",
    "WikiSource",
    "WikitextCache",
    "WikitextCleaner",
]
