"""Tests for the wiki client: caching, retries and error translation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.exceptions import ApplicationError
from app.ingestion import MediaWikiClient, WikiPageNotFoundError
from app.ingestion import wiki_client as wiki_client_module

PAGE_WIKITEXT = "== The Subway ==\n*NO CURE\n"


class WikiStub:
    """Answers ``action=parse`` requests and counts them."""

    def __init__(self, responses: list[httpx.Response] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.responses = responses or []
        self._served = 0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.responses:
            # A scripted stub repeats its last response, so a retry sees the
            # same failure instead of a healthy default.
            response = self.responses[min(self._served, len(self.responses) - 1)]
            self._served += 1
            return response
        page = request.url.params["page"]
        return httpx.Response(
            200,
            json={
                "parse": {
                    "title": page,
                    "wikitext": PAGE_WIKITEXT,
                }
            },
        )


def settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "ai_api_key": "sk-test",
        "wiki_base_url": "https://wiki.test",
        "wiki_cache_dir": tmp_path / "cache",
        "wiki_cache_enabled": True,
    }
    return Settings(**{**base, **overrides})


def build_client(stub: WikiStub, tmp_path: Path, **overrides: Any) -> MediaWikiClient:
    return MediaWikiClient(settings(tmp_path, **overrides), transport=stub.transport())


@pytest.fixture(autouse=True)
def _instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the retry test fast; the backoff itself is not what is under test."""
    monkeypatch.setattr(wiki_client_module, "BACKOFF_SECONDS", 0.0)


class TestFetching:
    async def test_returns_the_wikitext_of_a_page(self, tmp_path: Path) -> None:
        stub = WikiStub()
        client = build_client(stub, tmp_path)

        assert await client.fetch_page("Graffiti") == PAGE_WIKITEXT
        assert stub.requests[0].url.params["action"] == "parse"
        assert stub.requests[0].url.params["prop"] == "wikitext"

    async def test_sends_a_descriptive_user_agent(self, tmp_path: Path) -> None:
        stub = WikiStub()
        client = build_client(stub, tmp_path)

        await client.fetch_page("Graffiti")

        assert "User-Agent" in stub.requests[0].headers

    async def test_second_call_is_served_from_the_cache(self, tmp_path: Path) -> None:
        stub = WikiStub()
        client = build_client(stub, tmp_path)

        first = await client.fetch_page("Graffiti")
        second = await client.fetch_page("Graffiti")

        assert first == second
        assert len(stub.requests) == 1

    async def test_refresh_bypasses_the_cache(self, tmp_path: Path) -> None:
        stub = WikiStub()
        client = build_client(stub, tmp_path)

        await client.fetch_page("Graffiti")
        await client.fetch_page("Graffiti", refresh=True)

        assert len(stub.requests) == 2

    async def test_cache_can_be_disabled(self, tmp_path: Path) -> None:
        stub = WikiStub()
        client = build_client(stub, tmp_path, wiki_cache_enabled=False)

        await client.fetch_page("Graffiti")
        await client.fetch_page("Graffiti")

        assert len(stub.requests) == 2


class TestErrors:
    async def test_a_missing_page_raises_a_domain_error(self, tmp_path: Path) -> None:
        stub = WikiStub(
            [
                httpx.Response(
                    200,
                    json={"error": {"code": "missingtitle", "info": "page does not exist"}},
                )
            ]
        )
        client = build_client(stub, tmp_path)

        with pytest.raises(WikiPageNotFoundError):
            await client.fetch_page("Nope")

    async def test_an_empty_parse_payload_is_reported(self, tmp_path: Path) -> None:
        stub = WikiStub([httpx.Response(200, json={"parse": {"wikitext": "   "}})])
        client = build_client(stub, tmp_path)

        with pytest.raises(ApplicationError, match="Could not fetch"):
            await client.fetch_page("Graffiti")

    async def test_a_transient_failure_is_retried(self, tmp_path: Path) -> None:
        stub = WikiStub(
            [
                httpx.Response(503, text="upstream is having a moment"),
                httpx.Response(200, json={"parse": {"wikitext": PAGE_WIKITEXT}}),
            ]
        )
        client = build_client(stub, tmp_path)

        assert await client.fetch_page("Graffiti") == PAGE_WIKITEXT
        assert len(stub.requests) == 2

    async def test_a_permanent_failure_stops_after_the_budget(self, tmp_path: Path) -> None:
        stub = WikiStub([httpx.Response(500, text="boom") for _ in range(5)])
        client = build_client(stub, tmp_path)

        with pytest.raises(ApplicationError, match="Could not fetch"):
            await client.fetch_page("Graffiti")
        assert len(stub.requests) == wiki_client_module.MAX_ATTEMPTS

    async def test_a_non_json_reply_is_reported(self, tmp_path: Path) -> None:
        stub = WikiStub([httpx.Response(200, text="<html>not json</html>")])
        client = build_client(stub, tmp_path)

        with pytest.raises(ApplicationError, match="Could not fetch"):
            await client.fetch_page("Graffiti")


class TestBatchFetching:
    async def test_a_missing_page_does_not_abort_the_batch(self, tmp_path: Path) -> None:
        stub = WikiStub(
            [
                httpx.Response(200, json={"parse": {"wikitext": PAGE_WIKITEXT}}),
                httpx.Response(
                    200, json={"error": {"code": "missingtitle", "info": "gone"}}
                ),
                httpx.Response(200, json={"parse": {"wikitext": "== B ==\n*keep me"}}),
            ]
        )
        client = build_client(stub, tmp_path)

        pages = await client.fetch_pages(["A", "B", "C"])

        assert set(pages) == {"A", "C"}
        assert pages["C"] == "== B ==\n*keep me"


class TestPageUrl:
    def test_url_uses_underscores_and_escapes_the_title(self, tmp_path: Path) -> None:
        client = build_client(WikiStub(), tmp_path)

        assert (
            client.page_url("Ellis/Quotes and Captions")
            == "https://wiki.test/wiki/Ellis/Quotes_and_Captions"
        )


class TestCacheLayout:
    def test_each_page_gets_one_file_named_after_it(self, tmp_path: Path) -> None:
        client = build_client(WikiStub(), tmp_path)

        path = client._cache.path_for("Ellis/Quotes and Captions")

        assert path.parent == tmp_path / "cache"
        assert "/" not in path.name

    def test_cached_markup_is_plain_utf8(self, tmp_path: Path) -> None:
        client = build_client(WikiStub(), tmp_path)
        client._cache.write("Graffiti", "== Нет ==")

        assert client._cache.read("Graffiti") == "== Нет =="
        assert json.loads("{}") == {}
