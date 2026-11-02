"""Чтение сырого викитекста из вики Left 4 Dead на Fandom.

Клиент обращается к эндпоинту MediaWiki ``action=parse``, который возвращает
исходную разметку страницы — сырьё, с которым работают парсеры. Для
воспроизводимого конвейера важны два свойства:

* **Дисковый кэш.** Скачанный викитекст хранится в ``wiki_cache_dir``, поэтому
  повторный разбор не тратит сетевой обход, а корпус можно пересобрать офлайн.
* **Вежливые запросы.** Описательный ``User-Agent`` и ограниченные повторы с
  задержкой, потому что вики — чужой сервер.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.parse
from collections.abc import Sequence
from pathlib import Path

import httpx

from app.config import Settings
from app.exceptions import ApplicationError

logger = logging.getLogger("l4d.ingestion.wiki")

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 1.5
UNSAFE_FILENAME_PATTERN = re.compile(r"[^\w.-]+", re.UNICODE)
"""Символы, сохраняемые в имени файла кэша.

Корпус читается из русской вики, поэтому заголовки кириллические; шаблон
только из ASCII схлопнул бы каждый из них в один и тот же пустой слаг.
"""


class WikiPageNotFoundError(ApplicationError):
    """Возбуждается, когда на вики нет страницы с запрошенным заголовком."""

    def __init__(self, page: str) -> None:
        self.page = page
        super().__init__(f"Wiki page {page!r} does not exist")


class WikitextCache:
    """Кэш сырого викитекста: файл на страницу."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def path_for(self, page: str) -> Path:
        """Вернуть файл кэша, обслуживающий ``page``."""
        slug = UNSAFE_FILENAME_PATTERN.sub("_", page).strip("_")
        return self._directory / f"{slug}.wikitext"

    def read(self, page: str) -> str | None:
        """Вернуть закэшированную разметку ``page`` или ``None``, если её нет."""
        path = self.path_for(page)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def write(self, page: str, wikitext: str) -> None:
        """Сохранить разметку ``page``, создав каталог кэша при необходимости."""
        self._directory.mkdir(parents=True, exist_ok=True)
        self.path_for(page).write_text(wikitext, encoding="utf-8")


class MediaWikiClient:
    """Забирает разметку страниц, предпочитая локальный кэш сети."""

    def __init__(
        self,
        settings: Settings,
        *,
        cache: WikitextCache | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._cache = cache if cache is not None else WikitextCache(settings.wiki_cache_dir)
        self._client = httpx.AsyncClient(
            base_url=settings.wiki_base_url,
            timeout=settings.wiki_request_timeout_seconds,
            headers={"User-Agent": settings.wiki_user_agent, "Accept": "application/json"},
            follow_redirects=True,
            transport=transport,
        )

    def page_url(self, page: str) -> str:
        """Вернуть читаемый URL ``page``, сохраняемый как происхождение записи."""
        slug = urllib.parse.quote(page.replace(" ", "_"))
        return f"{self._settings.wiki_base_url}/wiki/{slug}"

    async def fetch_page(self, page: str, *, refresh: bool = False) -> str:
        """Вернуть викитекст ``page``.

        ``refresh`` обходит кэш и перезаписывает его свежей разметкой.
        """
        if self._settings.wiki_cache_enabled and not refresh:
            cached = self._cache.read(page)
            if cached is not None:
                logger.debug("Cache hit for %s", page)
                return cached

        wikitext = await self._request_wikitext(page)
        if self._settings.wiki_cache_enabled:
            self._cache.write(page, wikitext)
        return wikitext

    async def fetch_pages(self, pages: Sequence[str], *, refresh: bool = False) -> dict[str, str]:
        """Забрать несколько страниц последовательно, терпя отсутствующую.

        Страница, исчезнувшая с вики, не должна срывать всю сборку корпуса,
        поэтому сбой логируется, а страница пропускается.
        """
        collected: dict[str, str] = {}
        for page in pages:
            try:
                collected[page] = await self.fetch_page(page, refresh=refresh)
            except (WikiPageNotFoundError, httpx.HTTPError) as error:
                logger.warning("Skipping page %s: %s", page, error)
        return collected

    async def aclose(self) -> None:
        """Освободить пул HTTP-соединений."""
        await self._client.aclose()

    async def _request_wikitext(self, page: str) -> str:
        """Вызвать API MediaWiki, повторяя при временных сбоях."""
        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = await self._client.get(
                    "/api.php",
                    params={
                        "action": "parse",
                        "page": page,
                        "prop": "wikitext",
                        "format": "json",
                        "formatversion": "2",
                        "redirects": "1",
                    },
                )
                if response.status_code in RETRYABLE_STATUS_CODES:
                    raise httpx.HTTPStatusError(
                        f"Wiki returned {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return self._extract_wikitext(page, response.text)
            except (httpx.HTTPError, ValueError) as error:
                last_error = error
                if attempt == MAX_ATTEMPTS:
                    break
                delay = BACKOFF_SECONDS * attempt
                logger.warning(
                    "Wiki request for %s failed (attempt %d/%d), retrying in %.1fs: %s",
                    page,
                    attempt,
                    MAX_ATTEMPTS,
                    delay,
                    error,
                )
                await asyncio.sleep(delay)

        raise ApplicationError(f"Could not fetch wiki page {page!r}: {last_error}")

    @staticmethod
    def _extract_wikitext(page: str, payload: str) -> str:
        """Извлечь разметку из ответа ``action=parse``."""
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError(f"Wiki returned a non-JSON response: {error}") from error

        if "error" in document:
            code = document["error"].get("code")
            if code == "missingtitle":
                raise WikiPageNotFoundError(page)
            raise ValueError(f"Wiki API error: {document['error'].get('info')}")

        parse = document.get("parse")
        if not isinstance(parse, dict):
            raise WikiPageNotFoundError(page)

        wikitext = parse.get("wikitext")
        if not isinstance(wikitext, str) or not wikitext.strip():
            raise ValueError(f"Wiki returned empty markup for {page!r}")
        return wikitext
