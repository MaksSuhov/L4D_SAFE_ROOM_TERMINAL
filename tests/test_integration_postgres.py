"""Сквозные тесты против настоящего PostgreSQL + pgvector.

OpenAI API подменяется локальной HTTP-заглушкой, поэтому весь стек — роутеры,
внедрение зависимостей, сервис RAG, сборщик промпта, репозитории, поиск pgvector
и движок SQLAlchemy — работает по-настоящему, без доступа к сети и ключа API.

Модуль заводит собственную базу ``<database>_test`` и пропускается, когда сервер
PostgreSQL недоступен, что оставляет ``pytest`` зелёным на голой машине:

    docker compose up -d
    pytest tests/test_integration_postgres.py
"""

from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
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
from app.schemas import EmbeddedLoreChunk, LoreChunkCreate
from app.services import LoreIngestionService, OpenAIClient, content_hash
from app.taxonomy import LoreTopic, SourceType
from seed import DatabaseSeeder

EMBEDDING_SIZE = 1536
MOCK_ANSWER = "Aw man, a Tank? Get a wall behind you and light it up."
CHAT_PATH = "/chat"
CHARACTERS_PATH = "/characters"
HEALTH_PATH = "/health"

TANK_FACT = "Танк — Особый заражённый: у него около 4000 единиц здоровья."
WITCH_FACT = "Ведьма нападает на того, кто её потревожит."
WALL_RUMOUR = "На стене убежища написано, что это правительство нас травит."
VOICE_SAMPLE = "Эллис: Мы отличная команда, чувак!"


def unit_vector(axis: int, size: int = EMBEDDING_SIZE) -> list[float]:
    """Вернуть базисный вектор, нужный для детерминированных косинусных дистанций."""
    vector = [0.0] * size
    vector[axis] = 1.0
    return vector


def topic_embedding(text_value: str) -> list[float]:
    """Отобразить текст на базисный вектор, чтобы порядок выдачи был осмысленным.

    Тексты на одну тему попадают на одну ось, что даёт косинусную близость 1.0 для
    совпадающего запроса и 0.0 для постороннего. Оси разнесены далеко друг от друга,
    чтобы каждый уровень доказательств попадал на свою.
    """
    lowered = text_value.lower()
    if "tank" in lowered or "танк" in lowered:
        axis = 0
    elif "witch" in lowered or "ведьм" in lowered:
        axis = 1
    elif "safe room" in lowered or "убежищ" in lowered or "стене" in lowered:
        axis = 2
    elif "virus" in lowered or "вирус" in lowered:
        axis = 3
    else:
        axis = 4
    return unit_vector(axis)


def corpus() -> list[LoreChunkCreate]:
    """Миниатюрный корпус, хранящий по записи на каждый уровень доказательств."""
    return [
        LoreChunkCreate(
            content=TANK_FACT,
            source_type=SourceType.OFFICIAL,
            topic=LoreTopic.INFECTED,
            title="Танк › Обзор",
        ),
        LoreChunkCreate(
            content=WITCH_FACT,
            source_type=SourceType.OFFICIAL,
            topic=LoreTopic.INFECTED,
            title="Ведьма › Обзор",
        ),
        LoreChunkCreate(
            content=WALL_RUMOUR,
            source_type=SourceType.GRAFFITI,
            topic=LoreTopic.GRAFFITI,
            title="No Mercy › The Subway",
        ),
        LoreChunkCreate(
            content=VOICE_SAMPLE,
            source_type=SourceType.VOICE,
            topic=LoreTopic.QUOTES,
            title="Эллис · General",
            character_name="Эллис",
        ),
    ]


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    """Отвечает на эмбеддинги и чат-дополнения в сетевом формате OpenAI."""

    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 - name mandated by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append({"path": self.path, "payload": payload})

        if self.path.endswith("/embeddings"):
            sent = payload.get("input", "")
            texts = sent if isinstance(sent, list) else [sent]
            body: dict[str, Any] = {
                "object": "list",
                "model": payload.get("model", "text-embedding-3-small"),
                "data": [
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": topic_embedding(str(value)),
                    }
                    for index, value in enumerate(texts)
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
        """Заглушить стандартное логирование в stderr."""


class MockOpenAIServer:
    """Контекстный менеджер, поднимающий заглушку на свободном порту localhost."""

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
    """Вывести имя интеграционной базы из настроенного DSN."""
    url = make_url(base_url)
    return url.set(database=f"{url.database}_test").render_as_string(hide_password=False)


async def ensure_test_database(admin_url: str, database_name: str) -> None:
    """Создать интеграционную базу, если её ещё нет на сервере."""
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
    """Подготовить тестовую базу, наполнить её и отдать действующие настройки."""
    admin_url = get_settings().database_url
    integration_url = integration_database_url(admin_url)
    database_name = make_url(integration_url).database or ""

    try:
        await ensure_test_database(admin_url, database_name)
    except Exception as error:  # noqa: BLE001 - any driver or socket failure means "no server"
        pytest.skip(f"PostgreSQL is not reachable at {admin_url}: {error}")

    with MockOpenAIServer() as server:
        monkeypatch.setenv("DATABASE_URL", integration_url)
        monkeypatch.setenv("AI_API_KEY", "sk-mock-key")
        monkeypatch.setenv("AI_BASE_URL", server.base_url)
        monkeypatch.setenv("API_PREFIX", "/api/v1")
        # The stub answers every completion alike, so it cannot serve an expansion,
        # and the pipeline check below counts exactly one completion per chat turn.
        # HIDE itself is covered by the unit tests and by the HTTP test that injects
        # a stub expander.
        monkeypatch.setenv("HIDE_ENABLED", "false")
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
                await seeder.run(documents=corpus())
                await seeder.run(documents=corpus())  # second run must be a no-op

            yield settings
        finally:
            await database.drop_all()
            await ai_client.aclose()
            await database.dispose()

    get_settings.cache_clear()


@pytest.fixture
def mock_openai_server(seeded_settings: Settings) -> Iterator[list[dict[str, Any]]]:
    """Отдать запросы, записанные заглушкой во время проверяемого запроса."""
    yield _MockOpenAIHandler.requests


async def test_seed_creates_personas_lore_and_hnsw_index(seeded_settings: Settings) -> None:
    database = Database(seeded_settings)
    try:
        async with database.session() as session:
            characters = await CharacterRepository(session).get_all()
            lore_repository = LoreChunkRepository(session)

            assert [character.name for character in characters] == [
                "Эллис",
                "Ник",
                "Рошель",
                "Тренер",
                "Билл",
                "Фрэнсис",
                "Луис",
                "Зои",
            ]
            assert len(characters[0].few_shot_examples) == 2
            assert await lore_repository.count() == len(corpus())

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


async def test_seeding_twice_does_not_duplicate_records(seeded_settings: Settings) -> None:
    """Проверить, что при двойном наполнении записи не дублируются: фикстура уже
    запустила seed-скрипт дважды, а корпус должен сохраниться один раз.
    """
    database = Database(seeded_settings)
    try:
        async with database.session() as session:
            assert await LoreChunkRepository(session).count() == len(corpus())
    finally:
        await database.dispose()


async def test_personas_are_updated_in_place(seeded_settings: Settings) -> None:
    """Проверить, что правка персонажа применяется на месте, без пересоздания базы."""
    from seed import CHARACTER_SEEDS, DatabaseSeeder

    database = Database(seeded_settings)
    try:
        async with database.session() as session:
            characters = CharacterRepository(session)
            before = await characters.get_by_name("Эллис")
            assert before is not None
            before.system_prompt = "устаревшая персона"

            lore_repository = LoreChunkRepository(session)
            await DatabaseSeeder(
                database=database,
                character_repository=characters,
                lore_chunk_repository=lore_repository,
                lore_ingestion_service=LoreIngestionService(
                    lore_repository, OpenAIClient(seeded_settings)
                ),
            )._seed_characters()

        async with database.session() as session:
            refreshed = await CharacterRepository(session).get_by_name("Эллис")
            assert refreshed is not None
            assert refreshed.system_prompt == next(
                seed["system_prompt"] for seed in CHARACTER_SEEDS if seed["name"] == "Эллис"
            )
    finally:
        await database.dispose()


async def test_vector_search_orders_chunks_by_cosine_similarity(
    seeded_settings: Settings,
) -> None:
    database = Database(seeded_settings)
    try:
        async with database.session() as session:
            repository = LoreChunkRepository(session)
            # A topic no seeded record uses, so the crafted vectors are the only
            # candidates and their order is not disturbed by ties with the corpus.
            await repository.create_chunk(
                embedded_record("orthogonal to the query", unit_vector(1), isolated=True)
            )
            await repository.create_chunk(
                embedded_record("identical to the query", unit_vector(0), isolated=True)
            )

            matches = await repository.search_similar_chunks(
                unit_vector(0), limit=2, topic=str(LoreTopic.SAFE_ROOMS)
            )

            assert [match.chunk.content for match in matches] == [
                "identical to the query",
                "orthogonal to the query",
            ]
            assert matches[0].similarity == pytest.approx(1.0)
            assert matches[1].similarity == pytest.approx(0.0)

            official_only = await repository.search_similar_chunks(
                unit_vector(0), limit=10, source_types=[SourceType.OFFICIAL]
            )
            tiers = {match.chunk.source_type for match in official_only}
            assert tiers == {str(SourceType.OFFICIAL)}
    finally:
        await database.dispose()


async def test_vector_search_can_be_narrowed_to_one_character(
    seeded_settings: Settings,
) -> None:
    database = Database(seeded_settings)
    try:
        async with database.session() as session:
            repository = LoreChunkRepository(session)

            matches = await repository.search_similar_chunks(
                unit_vector(0), limit=10, character_name="Эллис"
            )
            assert [match.chunk.content for match in matches] == [VOICE_SAMPLE]

            nobody = await repository.search_similar_chunks(
                unit_vector(0), limit=10, character_name="Ник"
            )
            assert nobody == []
    finally:
        await database.dispose()


async def test_existing_hashes_report_what_is_stored(seeded_settings: Settings) -> None:
    database = Database(seeded_settings)
    try:
        async with database.session() as session:
            repository = LoreChunkRepository(session)

            found = await repository.existing_content_hashes(
                [content_hash(TANK_FACT), content_hash("текст, которого нет")]
            )

            assert found == {content_hash(TANK_FACT)}
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
    assert [character["name"] for character in characters.json()] == [
        "Эллис",
        "Ник",
        "Рошель",
        "Тренер",
        "Билл",
        "Фрэнсис",
        "Луис",
        "Зои",
    ]

    assert chat.status_code == 200
    payload = chat.json()
    assert payload["character_name"] == "Эллис"
    assert payload["answer"] == MOCK_ANSWER

    # The question mentions a Tank, so the Tank record of the official tier is
    # retrieved at full similarity and offered to the assistant as a fact.
    official = [source for source in payload["sources"] if source["source_type"] == "official"]
    assert official[0]["similarity"] == pytest.approx(1.0)
    assert official[0]["content"] == TANK_FACT
    assert official[0]["title"] == "Танк › Обзор"

    assert missing.status_code == 404

    chat_requests = [
        request for request in mock_openai_server if request["path"].endswith("/chat/completions")
    ]
    assert len(chat_requests) == 1
    messages = chat_requests[0]["payload"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("Ты — Эллис")
    # The configured language instruction is appended to the persona prompt, so
    # the answer is always Russian regardless of the question language.
    assert seeded_settings.answer_language_instruction in messages[0]["content"]
    assert [message["role"] for message in messages] == [
        "system",
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]

    brief = messages[1]["content"]
    assert "<official_lore>" in brief
    assert "<wall_graffiti>" in brief
    assert "<speech_samples>" in brief
    assert TANK_FACT in brief
    assert WALL_RUMOUR in brief
    assert VOICE_SAMPLE in brief
    # Tier separation: the wall inscription must never sit in the factual block.
    assert WALL_RUMOUR not in brief.split("<wall_graffiti>")[0]

    assert messages[-1] == {"role": "user", "content": "Tank inbound?"}


async def test_lore_chunk_round_trips_its_embedding(seeded_settings: Settings) -> None:
    database = Database(seeded_settings)
    try:
        async with database.session() as session:
            stored = await LoreChunkRepository(session).create_chunk(
                embedded_record("embedding round trip", unit_vector(7))
            )
            chunk_id = stored.id

        async with database.session() as session:
            reloaded = await session.get(LoreChunk, chunk_id)

            assert reloaded is not None
            assert reloaded.embedding[7] == pytest.approx(1.0)
            assert len(reloaded.embedding) == EMBEDDING_SIZE
    finally:
        await database.dispose()


def embedded_record(
    content: str,
    embedding: Sequence[float],
    *,
    isolated: bool = False,
) -> EmbeddedLoreChunk:
    """Собрать сохраняемую запись вокруг вектора, сделанного вручную.

    ``isolated`` помещает запись в тему, не занятую ни одной записью корпуса: так
    тест о порядке векторов не зависит от наполненных данных.
    """
    return EmbeddedLoreChunk(
        content=content,
        source_type=SourceType.OFFICIAL,
        topic=LoreTopic.SAFE_ROOMS if isolated else LoreTopic.INFECTED,
        title="Integration",
        content_hash=content_hash(content),
        embedding=list(embedding),
    )
