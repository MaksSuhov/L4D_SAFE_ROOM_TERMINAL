"""End-to-end tests against a real PostgreSQL + pgvector instance.

The OpenAI API is replaced by a local HTTP stub, so the whole stack — routers,
dependency injection, RAG service, repositories, pgvector search and the SQLAlchemy
engine — runs for real without network access or an API key.

The module provisions its own ``<database>_test`` database and is skipped when no
PostgreSQL server is reachable, which keeps ``pytest`` green on a bare machine:

    docker compose up -d
    pytest tests/test_integration_postgres.py
"""

from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterator, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import Settings, get_settings
from app.database import Database
from app.main import create_app, lifespan
from app.models import LoreChunk
from app.repositories import CharacterRepository, LoreChunkRepository
from app.services import LoreIngestionService, OpenAIClient
from seed import DatabaseSeeder

EMBEDDING_SIZE = 1536
MOCK_ANSWER = "Aw man, a Tank? Get a wall behind you and light it up."
CHAT_PATH = "/chat"
CHARACTERS_PATH = "/characters"
HEALTH_PATH = "/health"


def unit_vector(axis: int, size: int = EMBEDDING_SIZE) -> list[float]:
    """Return a basis vector, used to craft deterministic cosine distances."""
    vector = [0.0] * size
    vector[axis] = 1.0
    return vector


def topic_embedding(text_value: str) -> list[float]:
    """Map a text to a basis vector so that retrieval order is meaningful.

    Texts about the same subject land on the same axis, which gives a cosine
    similarity of 1.0 for a matching query and 0.0 for an unrelated one.
    """
    lowered = text_value.lower()
    if "tank" in lowered:
        axis = 0
    elif "witch" in lowered:
        axis = 1
    elif "safe room" in lowered:
        axis = 2
    else:
        axis = 3
    return unit_vector(axis)


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    """Answers embeddings and chat completions in the OpenAI wire format."""

    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 - name mandated by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append({"path": self.path, "payload": payload})

        if self.path.endswith("/embeddings"):
            body: dict[str, Any] = {
                "object": "list",
                "model": payload.get("model", "text-embedding-3-small"),
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": topic_embedding(str(payload.get("input", ""))),
                    }
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        elif self.path.endswith("/chat/completions"):
            body = {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 0,
                "model": payload.get("model", "gpt-4o-mini"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": MOCK_ANSWER},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        else:
            self.send_response(404)
            self.end_headers()
            return

        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        """Silence the default stderr logging."""


class MockOpenAIServer:
    """Context manager running the stub on a free localhost port."""

    def __init__(self) -> None:
        _MockOpenAIHandler.requests = []
        self._host = "127.0.0.1"
        self._server = ThreadingHTTPServer((self._host, 0), _MockOpenAIHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        port = int(self._server.server_address[1])
        return f"http://{self._host}:{port}/v1"

    @property
    def requests(self) -> list[dict[str, Any]]:
        return _MockOpenAIHandler.requests

    def __enter__(self) -> MockOpenAIServer:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def integration_database_url(base_url: str) -> str:
    """Derive the integration database name from the configured DSN."""
    url = make_url(base_url)
    return url.set(database=f"{url.database}_test").render_as_string(hide_password=False)


async def ensure_test_database(admin_url: str, database_name: str) -> None:
    """Create the integration database if the server does not have it yet."""
    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            exists = await connection.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": database_name},
            )
            if not exists:
                await connection.execute(text(f'CREATE DATABASE "{database_name}"'))
    finally:
        await engine.dispose()


@pytest.fixture
async def seeded_settings(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Settings]:
    """Provision the test database, seed it and expose the active settings."""
    admin_url = get_settings().database_url
    integration_url = integration_database_url(admin_url)
    database_name = make_url(integration_url).database or ""

    try:
        await ensure_test_database(admin_url, database_name)
    except Exception as error:  # noqa: BLE001 - any driver or socket failure means "no server"
        pytest.skip(f"PostgreSQL is not reachable at {admin_url}: {error}")

    with MockOpenAIServer() as server:
        monkeypatch.setenv("DATABASE_URL", integration_url)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key")
        monkeypatch.setenv("OPENAI_BASE_URL", server.base_url)
        monkeypatch.setenv("API_PREFIX", "/api/v1")
        get_settings.cache_clear()
        settings = get_settings()

        database = Database(settings)
        ai_client = OpenAIClient(settings)
        try:
            await database.drop_all()
            async with database.session() as session:
                lore_repository = LoreChunkRepository(session)
                seeder = DatabaseSeeder(
                    database=database,
                    character_repository=CharacterRepository(session),
                    lore_chunk_repository=lore_repository,
                    lore_ingestion_service=LoreIngestionService(lore_repository, ai_client),
                )
                await seeder.run()
                await seeder.run()  # second run must be a no-op

            yield settings
        finally:
            await database.drop_all()
            await ai_client.aclose()
            await database.dispose()

    get_settings.cache_clear()


@pytest.fixture
def mock_openai_server(seeded_settings: Settings) -> Iterator[list[dict[str, Any]]]:
    """Expose the requests recorded by the stub during the request under test."""
    yield _MockOpenAIHandler.requests


async def test_seed_creates_personas_lore_and_hnsw_index(seeded_settings: Settings) -> None:
    database = Database(seeded_settings)
    try:
        async with database.session() as session:
            characters = await CharacterRepository(session).get_all()
            lore_chunks = await LoreChunkRepository(session).count()

            assert [character.name for character in characters] == ["Ellis", "Bill"]
            assert len(characters[0].few_shot_examples) == 2
            assert lore_chunks == 6

            index_definition = await session.scalar(
                text(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE indexname = 'ix_lore_chunks_embedding_hnsw'"
                )
            )
            assert index_definition is not None
            assert "hnsw" in index_definition
            assert "vector_cosine_ops" in index_definition
    finally:
        await database.dispose()


async def test_vector_search_orders_chunks_by_cosine_similarity(
    seeded_settings: Settings,
) -> None:
    database = Database(seeded_settings)
    try:
        async with database.session() as session:
            repository = LoreChunkRepository(session)
            await repository.create_chunk(
                content="orthogonal to the query",
                data_type="integration",
                embedding=unit_vector(1),
            )
            await repository.create_chunk(
                content="identical to the query",
                data_type="integration",
                embedding=unit_vector(0),
            )

            matches = await repository.search_similar_chunks(
                unit_vector(0),
                limit=2,
                data_type="integration",
            )

            assert [match.chunk.content for match in matches] == [
                "identical to the query",
                "orthogonal to the query",
            ]
            assert matches[0].similarity == pytest.approx(1.0)
            assert matches[1].similarity == pytest.approx(0.0)

            filtered = await repository.search_similar_chunks(
                unit_vector(0),
                limit=10,
                data_type="environment",
            )
            assert len(filtered) == 1
            assert filtered[0].chunk.content.startswith("A safe room is the only")
    finally:
        await database.dispose()


async def test_chat_endpoint_runs_the_full_pipeline(
    seeded_settings: Settings,
    mock_openai_server: list[dict[str, Any]],
) -> None:
    application = create_app()

    async with lifespan(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get(f"{seeded_settings.api_prefix}{HEALTH_PATH}")
            characters = await client.get(f"{seeded_settings.api_prefix}{CHARACTERS_PATH}")
            chat = await client.post(
                f"{seeded_settings.api_prefix}{CHAT_PATH}",
                json={"character_id": characters.json()[0]["id"], "message": "Tank inbound?"},
            )
            missing = await client.post(
                f"{seeded_settings.api_prefix}{CHAT_PATH}",
                json={"character_id": 9999, "message": "Hello?"},
            )

    assert health.json()["database"] == "ok"
    assert [character["name"] for character in characters.json()] == ["Ellis", "Bill"]

    assert chat.status_code == 200
    payload = chat.json()
    assert payload["character_name"] == "Ellis"
    assert payload["answer"] == MOCK_ANSWER
    assert len(payload["sources"]) == seeded_settings.rag_top_k
    # The question mentions a Tank, so the two Tank records of the archive are
    # retrieved first and the assistant is given them as facts.
    assert payload["sources"][0]["similarity"] == pytest.approx(1.0)
    assert "The Tank is the rarest Special Infected" in payload["sources"][0]["content"]

    assert missing.status_code == 404

    chat_requests = [
        request for request in mock_openai_server if request["path"].endswith("/chat/completions")
    ]
    assert len(chat_requests) == 1
    messages = chat_requests[0]["payload"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("You are Ellis")
    assert [message["role"] for message in messages] == [
        "system",
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]

    archive_message = messages[1]["content"]
    assert "<archive_records>" in archive_message
    for position, source in enumerate(payload["sources"], start=1):
        assert f"{position}. [{source['data_type']}] {source['content']}" in archive_message

    assert messages[-1] == {"role": "user", "content": "Tank inbound?"}


async def test_lore_chunk_round_trips_its_embedding(seeded_settings: Settings) -> None:
    database = Database(seeded_settings)
    try:
        async with database.session() as session:
            stored = await LoreChunkRepository(session).create_chunk(
                content="embedding round trip",
                data_type="monster",
                embedding=unit_vector(7),
            )
            chunk_id = stored.id

        async with database.session() as session:
            reloaded = await session.get(LoreChunk, chunk_id)

            assert reloaded is not None
            assert reloaded.embedding[7] == pytest.approx(1.0)
            assert len(reloaded.embedding) == EMBEDDING_SIZE
    finally:
        await database.dispose()
