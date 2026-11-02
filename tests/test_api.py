"""Интеграционные тесты HTTP-слоя.

Подменяются только внешние адаптеры; настоящие роутеры, граф зависимостей и
сервис RAG работают по-настоящему, что и проверяет саму связку.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import (
    get_ai_client,
    get_character_repository,
    get_lore_chunk_repository,
    get_query_expander,
)
from app.interfaces import RetrievedChunk
from app.main import create_app
from app.taxonomy import SourceType
from tests.fakes import (
    FakeAIClient,
    FakeQueryExpander,
    InMemoryCharacterRepository,
    InMemoryLoreChunkRepository,
    build_character,
    build_chunk,
)

API_PREFIX = get_settings().api_prefix
ANSWER = "Aw man, the Tank? Keep a wall behind you."
HYPOTHETICAL = "Оружие в кампании «Вымерший центр»"


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
def query_expander() -> FakeQueryExpander:
    return FakeQueryExpander(hypotheticals=[HYPOTHETICAL])


@pytest.fixture
def client(
    character_repository: InMemoryCharacterRepository,
    lore_chunk_repository: InMemoryLoreChunkRepository,
    ai_client: FakeAIClient,
    query_expander: FakeQueryExpander,
) -> Iterator[TestClient]:
    application = create_app()
    application.dependency_overrides[get_character_repository] = lambda: character_repository
    application.dependency_overrides[get_lore_chunk_repository] = lambda: lore_chunk_repository
    application.dependency_overrides[get_ai_client] = lambda: ai_client
    application.dependency_overrides[get_query_expander] = lambda: query_expander

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
    """Проверить, что связка передаёт провайдеру двухуровневый бриф, а не плоский
    список записей.
    """
    client.post(
        f"{API_PREFIX}/chat",
        json={"character_id": 1, "message": "How do we kill a Tank?"},
    )

    context = str(ai_client.generation_calls[0]["context"])
    assert "<official_lore>" in context
    assert "<wall_graffiti>" in context
    assert "<speech_samples>" in context


def test_chat_embeds_the_hypothetical_formulations_with_the_question(
    client: TestClient,
    ai_client: FakeAIClient,
    query_expander: FakeQueryExpander,
) -> None:
    """Проверить, что расширение вопроса доходит до провайдера через весь HTTP-слой
    и стоит одного пакетного запроса на эмбеддинги.
    """
    client.post(
        f"{API_PREFIX}/chat",
        json={"character_id": 1, "message": "How do we kill a Tank?"},
    )

    assert query_expander.questions == ["How do we kill a Tank?"]
    assert ai_client.embedding_calls == [["How do we kill a Tank?", HYPOTHETICAL]]


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
    """Проверить, что предварительный запрос CORS пропускает origin фронтенда: он
    работает на собственном dev-сервере.
    """
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
