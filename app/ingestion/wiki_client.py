"""Reading raw wikitext from the Left 4 Dead Fandom wiki.

The client talks to the MediaWiki ``action=parse`` endpoint, which returns the
source markup of a page — the raw material the parsers work on. Two properties
matter for a reproducible pipeline:

* **A disk cache.** Donated wikitext is stored under ``wiki_cache_dir`` so a
  re-parse costs no network round trip, and the corpus can be rebuilt offline.
* **Polite requests.** A descriptive ``User-Agent`` and bounded retries with
  backoff, because a wiki is somebody else's server.
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
"""Characters kept in a cache filename.

The corpus is read from the Russian wiki, so titles are Cyrillic; an ASCII-only
pattern would collapse every one of them to the same empty slug.
"""


class WikiPageNotFoundError(ApplicationError):
    """Raised when the wiki has no page with the requested title."""

    def __init__(self, page: str) -> None:
        self.page = page
        super().__init__(f"Wiki page {page!r} does not exist")


class WikitextCache:
    """File-per-page cache of raw wikitext."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def path_for(self, page: str) -> Path:
        """Return the cache file backing ``page``."""
        slug = UNSAFE_FILENAME_PATTERN.sub("_", page).strip("_")
        return self._directory / f"{slug}.wikitext"

    def read(self, page: str) -> str | None:
        """Return the cached markup of ``page``, or ``None`` when absent."""
        path = self.path_for(page)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def write(self, page: str, wikitext: str) -> None:
        """Store the markup of ``page``, creating the cache directory on demand."""
        self._directory.mkdir(parents=True, exist_ok=True)
        self.path_for(page).write_text(wikitext, encoding="utf-8")


class MediaWikiClient:
    """Fetches page markup, preferring the local cache over the network."""

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
        """Return the readable URL of ``page``, stored as the provenance of a record."""
        slug = urllib.parse.quote(page.replace(" ", "_"))
        return f"{self._settings.wiki_base_url}/wiki/{slug}"

    async def fetch_page(self, page: str, *, refresh: bool = False) -> str:
        """Return the wikitext of ``page``.

        ``refresh`` bypasses the cache and rewrites it with fresh markup.
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
        """Fetch several pages sequentially, tolerating a missing one.

        A page that vanished from the wiki must not abort the whole corpus
        build, so the failure is logged and the page is left out.
        """
        collected: dict[str, str] = {}
        for page in pages:
            try:
                collected[page] = await self.fetch_page(page, refresh=refresh)
            except (WikiPageNotFoundError, httpx.HTTPError) as error:
                logger.warning("Skipping page %s: %s", page, error)
        return collected

    async def aclose(self) -> None:
        """Release the HTTP connection pool."""
        await self._client.aclose()

    async def _request_wikitext(self, page: str) -> str:
        """Call the MediaWiki API, retrying transient failures."""
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
        """Pull the markup out of an ``action=parse`` response."""
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
