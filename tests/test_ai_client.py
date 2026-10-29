"""Tests for the provider adapter: batching, ordering, translation fallback.

The adapter talks to :class:`~tests.provider_server.ProviderServer`, a local stub
speaking the OpenAI wire format, so request bodies are exercised for real without
a network or an API key.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.config import Settings
from app.exceptions import AIServiceError
from app.services import OpenAIClient
from tests.provider_server import ProviderServer


def build_client(server: ProviderServer, **overrides: Any) -> OpenAIClient:
    """Point the adapter at the stub, overriding the tunables a test cares about."""
    settings = Settings(
        ai_api_key="sk-test",
        ai_base_url=server.base_url,
        embedding_dimensions=overrides.pop("embedding_dimensions", 3),
        ingestion_batch_size=overrides.pop("ingestion_batch_size", 2),
        translation_batch_size=overrides.pop("translation_batch_size", 2),
        chat_max_tokens=overrides.pop("chat_max_tokens", 32),
        **overrides,
    )
    return OpenAIClient(settings)


class TestEmbeddings:
    async def test_splits_the_work_into_batches(self) -> None:
        with ProviderServer() as server:
            client = build_client(server)

            vectors = await client.get_embeddings(["a", "b", "c", "d", "e"])

            assert len(vectors) == 5
            batch_sizes = [
                len(request["payload"]["input"]) for request in server.embedding_requests()
            ]
            assert batch_sizes == [2, 2, 1]
            await client.aclose()

    async def test_restores_the_requested_order(self) -> None:
        """The API may answer out of order and restarts its index per batch."""
        with ProviderServer() as server:
            client = build_client(server)

            vectors = await client.get_embeddings(["a", "b", "c"])

            assert [vector[0] for vector in vectors] == [97.0, 98.0, 99.0]
            await client.aclose()

    async def test_single_text_still_goes_through_the_batch_path(self) -> None:
        with ProviderServer() as server:
            client = build_client(server)

            vector = await client.get_embedding("hello")

            assert vector[0] == float(ord("h"))
            assert len(server.embedding_requests()) == 1
            await client.aclose()

    async def test_no_texts_means_no_request(self) -> None:
        with ProviderServer() as server:
            client = build_client(server)

            assert await client.get_embeddings([]) == []
            assert server.embedding_requests() == []
            await client.aclose()

    async def test_a_short_batch_is_rejected(self) -> None:
        """Fewer vectors than texts must fail loudly rather than shift the corpus."""
        with ProviderServer(drop_last_embedding=True) as server:
            client = build_client(server)

            with pytest.raises(AIServiceError, match="for a batch of"):
                await client.get_embeddings(["a", "b"])
            await client.aclose()

    async def test_a_surprise_vector_width_is_rejected(self) -> None:
        with ProviderServer(embedding_dimensions=7) as server:
            client = build_client(server, embedding_dimensions=3)

            with pytest.raises(AIServiceError, match="dimensions"):
                await client.get_embedding("hello")
            await client.aclose()


class TestTranslation:
    async def test_translates_every_batch(self) -> None:
        with ProviderServer(chat_replies=['["один", "два"]', '["три"]']) as server:
            client = build_client(server)

            translations = await client.translate_texts(["one", "two", "three"])

            assert translations == ["один", "два", "три"]
            assert len(server.chat_requests()) == 2
            await client.aclose()

    async def test_translation_is_deterministic(self) -> None:
        with ProviderServer(chat_replies=['["один"]']) as server:
            client = build_client(server)

            await client.translate_texts(["one"])

            assert server.chat_requests()[0]["payload"]["temperature"] == 0.0
            await client.aclose()

    async def test_tolerates_prose_around_the_json_array(self) -> None:
        with ProviderServer(chat_replies=['Вот перевод:\n["один"]\nГотово.']) as server:
            client = build_client(server)

            assert await client.translate_texts(["one"]) == ["один"]
            await client.aclose()

    async def test_a_malformed_batch_falls_back_to_single_texts(self) -> None:
        """A corpus must never be lost because one batch came back wrong."""
        with ProviderServer(chat_replies=['["один"]', '["один"]', '["два"]']) as server:
            client = build_client(server)

            assert await client.translate_texts(["one", "two"]) == ["один", "два"]
            await client.aclose()

    async def test_a_failed_text_keeps_its_original_wording(self) -> None:
        with ProviderServer(chat_replies=["not json at all", "not json either"]) as server:
            client = build_client(server)

            assert await client.translate_texts(["one", "two"]) == ["one", "two"]
            await client.aclose()

    async def test_no_texts_means_no_request(self) -> None:
        with ProviderServer() as server:
            client = build_client(server)

            assert await client.translate_texts([]) == []
            assert server.chat_requests() == []
            await client.aclose()
