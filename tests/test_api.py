"""Integration tests for the HTTP layer.

Only the outermost adapters are replaced; the real routers, dependency graph and
RAG service are exercised, which is what verifies the wiring itself.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import get_ai_client, get_character_repository, get_lore_chunk_repository
from app.interfaces import RetrievedChunk
from app.main import create_app
from app.taxonomy import SourceType
from tests.fakes import (
    FakeAIClient,
    InMemoryCharacterRepository,
    InMemoryLoreChunkRepository,
    build_character,
    build_chunk,
)

API_PREFIX = get_settings().api_prefix
ANSWER = "Aw man, the Tank? Keep a wall behind you."


@pytest.fixture
def character_repository() -> InMemoryCharacterRepository:
    return InMemoryCharacterRepository([build_character(character_id=1, name="Эллис")])


@pytest.fixture
def lore_chunk_repository() -> InMemoryLoreChunkRepository:
    return InMemoryLoreChunkRepository(
        [
            RetrievedChunk(
                build_chunk(
                    chunk_id=7,
                    content="Tank lore",
                    source_type=SourceType.OFFICIAL,
                    title="Tank › Overview",
                ),
                0.88,
            )
        ]
    )


@pytest.fixture
def ai_client() -> FakeAIClient:
    return FakeAIClient(answer=ANSWER)


@pytest.fixture
def client(
    character_repository: InMemoryCharacterRepository,
    lore_chunk_repository: InMemoryLoreChunkRepository,
    ai_client: FakeAIClient,
) -> Iterator[TestClient]:
    application = create_app()
    application.dependency_overrides[get_character_repository] = lambda: character_repository
    application.dependency_overrides[get_lore_chunk_repository] = lambda: lore_chunk_repository
    application.dependency_overrides[get_ai_client] = lambda: ai_client

    with TestClient(application) as test_client:
        yield test_client

    application.dependency_overrides.clear()


def test_chat_returns_answer_with_retrieved_sources(client: TestClient) -> None:
    response = client.post(
        f"{API_PREFIX}/chat",
        json={"character_id": 1, "message": "How do we kill a Tank?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["character_id"] == 1
    assert payload["character_name"] == "Эллис"
    assert payload["answer"] == ANSWER
    assert payload["model"] == "fake-model"
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["id"] == 7
    assert payload["sources"][0]["similarity"] == 0.88
    assert payload["sources"][0]["source_type"] == "official"
    assert payload["sources"][0]["topic"] == "infected"
    assert payload["sources"][0]["title"] == "Tank › Overview"


def test_chat_hands_the_two_level_brief_to_the_provider(
    client: TestClient,
    ai_client: FakeAIClient,
) -> None:
    """The wiring must deliver the tiered brief, not a flat list of records."""
    client.post(
        f"{API_PREFIX}/chat",
        json={"character_id": 1, "message": "How do we kill a Tank?"},
    )

    context = str(ai_client.generation_calls[0]["context"])
    assert "<official_lore>" in context
    assert "<wall_graffiti>" in context
    assert "<speech_samples>" in context


def test_chat_with_unknown_character_returns_not_found(client: TestClient) -> None:
    response = client.post(
        f"{API_PREFIX}/chat",
        json={"character_id": 999, "message": "Anyone home?"},
    )

    assert response.status_code == 404
    assert "999" in response.json()["detail"]


@pytest.mark.parametrize("message", ["", "   ", "x" * 2001])
def test_chat_rejects_invalid_messages(client: TestClient, message: str) -> None:
    response = client.post(
        f"{API_PREFIX}/chat",
        json={"character_id": 1, "message": message},
    )

    assert response.status_code == 422


def test_chat_requires_a_positive_character_id(client: TestClient) -> None:
    response = client.post(
        f"{API_PREFIX}/chat",
        json={"character_id": 0, "message": "Hello?"},
    )

    assert response.status_code == 422


def test_characters_endpoint_lists_personas(client: TestClient) -> None:
    response = client.get(f"{API_PREFIX}/characters")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": 1,
            "name": "Эллис",
            "system_prompt": "Ты — Эллис.",
            "few_shot_examples": [],
        }
    ]


def test_health_endpoint_reports_liveness(client: TestClient) -> None:
    response = client.get(f"{API_PREFIX}/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == get_settings().app_version


def test_cors_preflight_allows_the_frontend_origin(client: TestClient) -> None:
    """The bundled frontend runs on its own dev server, so it needs CORS."""
    response = client.options(
        f"{API_PREFIX}/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "POST" in response.headers["access-control-allow-methods"]


def test_cors_does_not_authorise_unknown_origins(client: TestClient) -> None:
    response = client.get(
        f"{API_PREFIX}/characters",
        headers={"Origin": "http://not-the-terminal.example"},
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_openapi_documents_both_endpoints(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert f"{API_PREFIX}/chat" in schema["paths"]
    assert f"{API_PREFIX}/characters" in schema["paths"]
