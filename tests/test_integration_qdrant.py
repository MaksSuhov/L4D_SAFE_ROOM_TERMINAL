"""Сквозные тесты против Qdrant в локальном режиме.

Хранилище поднимается в процессе (``AsyncQdrantClient(":memory:")``), провайдер
подменяется локальной HTTP-заглушкой, а модель BM25 — детерминированным двойником,
поэтому весь стек работает по-настоящему: роутеры, граф зависимостей, сервис RAG,
сборщик промпта, репозитории, гибридный поиск и наполнение архива. Ни сети, ни
ключа API, ни контейнера для этого не требуется.

Локальный Qdrant не применяет payload-индексы (и предупреждает об этом), но
фильтры выполняет, поэтому проверки остаются честными; отключается только
предупреждение.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import httpx
import pytest
from qdrant_client import AsyncQdrantClient

from app.config import Settings, get_settings
from app.dependencies import get_lore_chunk_repository, get_vector_store
from app.main import create_app, lifespan
from app.schemas import EmbeddedLoreChunk, LoreChunkCreate
from app.services import LoreIngestionService, OpenAIClient, content_hash
from app.taxonomy import LoreTopic, SourceType
from app.vector import (
    QdrantCharacterRepository,
    QdrantLoreChunkRepository,
    VectorStore,
)
from app.vector.schema import ensure_lore_collection
from seed import CHARACTER_SEEDS, TerminalSeeder
from tests.fakes import FakeAIClient, FakeSparseEncoder
from tests.provider_server import ProviderServer

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

EMBEDDING_SIZE = 5
MOCK_ANSWER = "Aw man, a Tank? Get a wall behind you and light it up."
COLLECTION = "lore_chunks"

CHAT_PATH = "/chat"
CHARACTERS_PATH = "/characters"
HEALTH_PATH = "/health"

TANK_FACT = "Танк — Особый заражённый: у него около 4000 единиц здоровья."
WITCH_FACT = "Ведьма нападает на того, кто её потревожит."
WALL_RUMOUR = "На стене убежища написано, что это правительство нас травит."
VOICE_SAMPLE = "Эллис: Мы отличная команда, чувак!"

PERSONA_NAMES = [
    "Эллис",
    "Ник",
    "Рошель",
    "Тренер",
    "Билл",
    "Фрэнсис",
    "Луис",
    "Зои",
]


def unit_vector(axis: int, size: int = EMBEDDING_SIZE) -> list[float]:
    """Вернуть базисный вектор, нужный для детерминированных косинусных дистанций."""
    vector = [0.0] * size
    vector[axis] = 1.0
    return vector


def topic_embedding(text: str) -> list[float]:
    """Отобразить текст на базисный вектор, чтобы порядок выдачи был осмысленным.

    Тексты одной темы попадают на одну ось, что даёт косинусную близость 1.0 для
    совпадающего запроса и 0.0 для постороннего. Оси разнесены по темам, поэтому
    каждый уровень доказательств попадает на свою.
    """
    lowered = text.lower()
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


def embedded_record(
    content: str,
    embedding: Sequence[float],
    *,
    title: str,
    source_type: SourceType = SourceType.OFFICIAL,
    topic: LoreTopic = LoreTopic.INFECTED,
) -> EmbeddedLoreChunk:
    """Собрать сохраняемую запись вокруг вектора, сделанного вручную."""
    return EmbeddedLoreChunk(
        content=content,
        source_type=source_type,
        topic=topic,
        title=title,
        content_hash=content_hash(content),
        embedding=list(embedding),
    )


@dataclass
class SeededTerminal:
    """Наполненный терминал вместе с собранным графом его зависимостей."""

    settings: Settings
    store: VectorStore
    client: AsyncQdrantClient
    characters: QdrantCharacterRepository
    lore: QdrantLoreChunkRepository
    sparse: FakeSparseEncoder
    provider: ProviderServer


@pytest.fixture
async def seeded(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[SeededTerminal]:
    """Поднять заглушку провайдера, наполнить хранилище и отдать собранный граф."""
    with ProviderServer(
        embedding_dimensions=EMBEDDING_SIZE,
        embedding_encoder=topic_embedding,
        chat_replies=[MOCK_ANSWER],
    ) as provider:
        monkeypatch.setenv("AI_API_KEY", "sk-mock-key")
        monkeypatch.setenv("AI_BASE_URL", provider.base_url)
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", str(EMBEDDING_SIZE))
        # The stub answers every completion alike, so it cannot serve an expansion,
        # and the pipeline check below counts exactly one completion per chat turn.
        # HIDE itself is covered by unit tests and by the HTTP test that injects a
        # stub expander.
        monkeypatch.setenv("HIDE_ENABLED", "false")
        get_settings.cache_clear()
        settings = get_settings()

        client = AsyncQdrantClient(":memory:")
        store = VectorStore(settings, client=client)
        sparse = FakeSparseEncoder()
        lore = QdrantLoreChunkRepository(
            client,
            store.lore_collection,
            sparse,
            hybrid_enabled=settings.rag_hybrid_enabled,
            prefetch_multiplier=settings.rag_prefetch_multiplier,
        )
        characters = QdrantCharacterRepository(client, store.character_collection)
        ai_client = OpenAIClient(settings)

        try:
            seeder = TerminalSeeder(
                vector_store=store,
                character_repository=characters,
                lore_chunk_repository=lore,
                lore_ingestion_service=LoreIngestionService(lore, ai_client),
            )
            await seeder.run(documents=corpus())
            await seeder.run(documents=corpus())  # second run must be a no-op

            yield SeededTerminal(
                settings=settings,
                store=store,
                client=client,
                characters=characters,
                lore=lore,
                sparse=sparse,
                provider=provider,
            )
        finally:
            await ai_client.aclose()
            await client.close()
            get_settings.cache_clear()


async def test_seeding_writes_personas_and_the_archive(seeded: SeededTerminal) -> None:
    characters = await seeded.characters.get_all()

    assert [character.name for character in characters] == PERSONA_NAMES
    assert len(characters[0].few_shot_examples) == 2
    assert [character.id for character in characters] == list(range(1, len(PERSONA_NAMES) + 1))
    assert await seeded.lore.count() == len(corpus())


async def test_seeding_twice_does_not_duplicate_records(seeded: SeededTerminal) -> None:
    """Проверить, что при двойном наполнении записи не дублируются: фикстура уже
    запустила seed-скрипт дважды, а корпус должен сохраниться один раз.
    """
    assert await seeded.lore.count() == len(corpus())


async def test_personas_are_updated_in_place(seeded: SeededTerminal) -> None:
    """Проверить, что правка образа применяется на месте, без пересоздания хранилища."""
    before = await seeded.characters.get_by_name("Эллис")
    assert before is not None
    before.system_prompt = "устаревшая персона"
    await seeded.characters.save(before)

    await TerminalSeeder(
        vector_store=seeded.store,
        character_repository=seeded.characters,
        lore_chunk_repository=seeded.lore,
        lore_ingestion_service=LoreIngestionService(seeded.lore, FakeAIClient()),
    )._seed_characters()

    refreshed = await seeded.characters.get_by_name("Эллис")
    assert refreshed is not None
    assert refreshed.id == before.id
    assert refreshed.system_prompt == next(
        seed["system_prompt"] for seed in CHARACTER_SEEDS if seed["name"] == "Эллис"
    )


async def test_search_orders_chunks_by_cosine_similarity(seeded: SeededTerminal) -> None:
    identical = "одинаковый с запросом"
    orthogonal = "ортогональный запросу"

    await seeded.lore.create_chunk(
        embedded_record(identical, unit_vector(0), title="Identical", topic=LoreTopic.SAFE_ROOMS)
    )
    await seeded.lore.create_chunk(
        embedded_record(orthogonal, unit_vector(1), title="Orthogonal", topic=LoreTopic.SAFE_ROOMS)
    )

    matches = await seeded.lore.search_similar_chunks(
        unit_vector(0),
        "одинаковый",
        limit=2,
        topic=str(LoreTopic.SAFE_ROOMS),
    )

    assert [match.chunk.content for match in matches] == [identical, orthogonal]
    assert matches[0].similarity == pytest.approx(1.0)
    assert matches[1].similarity == pytest.approx(0.0)

    official_only = await seeded.lore.search_similar_chunks(
        unit_vector(0), "любой", limit=10, source_types=[SourceType.OFFICIAL]
    )
    assert {match.chunk.source_type for match in official_only} == {str(SourceType.OFFICIAL)}


async def test_search_can_be_narrowed_to_one_character(seeded: SeededTerminal) -> None:
    matches = await seeded.lore.search_similar_chunks(
        unit_vector(4), "команда", limit=10, character_name="Эллис"
    )
    assert [match.chunk.content for match in matches] == [VOICE_SAMPLE]

    nobody = await seeded.lore.search_similar_chunks(
        unit_vector(4), "команда", limit=10, character_name="Ник"
    )
    assert nobody == []


async def test_existing_hashes_report_what_is_stored(seeded: SeededTerminal) -> None:
    found = await seeded.lore.existing_content_hashes(
        [content_hash(TANK_FACT), content_hash("текст, которого нет")]
    )

    assert found == {content_hash(TANK_FACT)}


async def test_the_lexical_leg_changes_which_records_take_the_budget() -> None:
    """Лексическое плечо вправе поднять запись, которую плотное поставило последней.

    Корпус устроен так, что вопрос далёк по эмбеддингу от всех записей, но делит
    с одной из них редкое слово. Плотное плечо находит четыре записи и ставит
    нужную последней; лексическое находит только её и ставит первой. Слияние
    рангов выносит её в бюджет уровня, а близость при ней остаётся косинусной —
    то есть пороги уровней продолжают работать.
    """
    target_title = "Проект «Ноев ковчег»"
    decoy_titles = ["Танк › Обзор", "Ведьма › Обзор", "Курт › Обзор"]
    query_text = "Что известно про Ноев ковчег?"

    client = AsyncQdrantClient(":memory:")
    try:
        await ensure_lore_collection(client, COLLECTION, dimensions=EMBEDDING_SIZE)
        encoder = FakeSparseEncoder()
        hybrid = QdrantLoreChunkRepository(
            client, COLLECTION, encoder, hybrid_enabled=True, prefetch_multiplier=4
        )
        dense_only = QdrantLoreChunkRepository(client, COLLECTION, encoder, hybrid_enabled=False)

        await hybrid.create_chunks(
            [
                embedded_record(
                    "Проект «Ноев ковчег» упоминается в записях CEDA.",
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    title=target_title,
                ),
                embedded_record(
                    "Танк бьёт кулаком.",
                    [1.0, 0.1, 0.0, 0.0, 0.0],
                    title=decoy_titles[0],
                ),
                embedded_record(
                    "Ведьма плачет в углу.",
                    [1.0, 0.5, 0.0, 0.0, 0.0],
                    title=decoy_titles[1],
                ),
                embedded_record(
                    "Курт помогает выжившим.",
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    title=decoy_titles[2],
                ),
            ]
        )

        query_embedding = unit_vector(0)
        hybrid_matches = await hybrid.search_similar_chunks(
            query_embedding, query_text, limit=1, source_types=[SourceType.OFFICIAL]
        )
        dense_matches = await dense_only.search_similar_chunks(
            query_embedding, query_text, limit=1, source_types=[SourceType.OFFICIAL]
        )

        assert [match.chunk.title for match in hybrid_matches] == [target_title]
        assert [match.chunk.title for match in dense_matches] == [decoy_titles[0]]
        # Стоимость гибрида: бюджет заняла запись с нулевой косинусной близостью,
        # и вызывающий вправе её отбросить — так пороги уровней и поступают.
        assert hybrid_matches[0].similarity == pytest.approx(0.0)
    finally:
        await client.close()


async def test_chat_endpoint_runs_the_full_pipeline(seeded: SeededTerminal) -> None:
    application = create_app(seeded.settings)
    application.dependency_overrides[get_vector_store] = lambda: seeded.store
    application.dependency_overrides[get_lore_chunk_repository] = lambda: seeded.lore
    prefix = seeded.settings.api_prefix

    async with lifespan(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get(f"{prefix}{HEALTH_PATH}")
            characters = await client.get(f"{prefix}{CHARACTERS_PATH}")
            chat = await client.post(
                f"{prefix}{CHAT_PATH}",
                json={"character_id": characters.json()[0]["id"], "message": "Tank inbound?"},
            )
            missing = await client.post(
                f"{prefix}{CHAT_PATH}",
                json={"character_id": 9999, "message": "Hello?"},
            )

    assert health.json()["vector_store"] == "ok"
    assert [character["name"] for character in characters.json()] == PERSONA_NAMES

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

    chat_requests = seeded.provider.chat_requests()
    assert len(chat_requests) == 1
    messages = chat_requests[0]["payload"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("Ты — Эллис")
    # The configured language instruction is appended to the persona prompt, so
    # the answer is always Russian regardless of the question language.
    assert seeded.settings.answer_language_instruction in messages[0]["content"]
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


async def test_the_lexical_leg_is_asked_with_the_raw_question(seeded: SeededTerminal) -> None:
    """Проверить, что вопрос доходит до лексического плеча дословно.

    Косинусное плечо работает по вектору, лексическое — по тексту, и вернуть
    текст из эмбеддинга нельзя. Тест пришпиливает, что вопрос передаётся, и что
    передаётся именно он, а не какая-нибудь нормализованная форма.
    """
    await seeded.lore.search_similar_chunks(unit_vector(0), "Танк идёт!", limit=1)

    assert seeded.sparse.encoded_queries == ["Танк идёт!"]
