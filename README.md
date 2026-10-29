# Left 4 Dead Interactive Safe-Room Terminal

MVP-сервис на FastAPI: интерактивный терминал безопасной комнаты из Left 4 Dead. Вы выбираете выжившего
(Эллис, Билл и т.д), задаёте вопрос, а сервис отвечает **в образе персонажа**, опираясь на архив лора из
PostgreSQL + pgvector (RAG: retrieval-augmented generation).

Проект собран как демонстрация слоистой архитектуры: чистое разделение ответственности, SOLID,
инверсия зависимостей через `Protocol`, полная асинхронность и type hinting.

---

## Стек

| Слой | Технологии |
|---|---|
| API | FastAPI, Pydantic v2, Uvicorn |
| Бизнес-логика | чистый Python (`services/`), DI через конструкторы |
| Доступ к данным | SQLAlchemy 2.0 (`Mapped` / `mapped_column`), `AsyncSession`, asyncpg |
| Векторный поиск | PostgreSQL 16 + pgvector (косинусное расстояние, HNSW-индекс) |
| Внешний AI | OpenAI SDK: `text-embedding-3-small` (эмбеддинги), `gpt-4o-mini` (ответы) |

---

## Архитектура

```
L4D_SAFE_ROOM_TERMINAL/
├── app/
│   ├── config.py         # 1. Конфигурация: pydantic-settings, все настройки из ENV/.env
│   ├── database.py       # 1. БД: Base, асинхронный engine, фабрика сессий, Database
│   ├── models/           # 2. ORM-модели, по файлу на агрегат
│   │   ├── character_model.py    #    Character (+ JSONB few-shot примеры)
│   │   └── lore_chunk_model.py   #    LoreChunk (+ Vector(1536), HNSW-индекс)
│   ├── schemas/          # 2. Pydantic-схемы, сгруппированные по агрегатам
│   │   ├── character_schema.py   #    CharacterRead/Create/Base, FewShotExample
│   │   ├── lore_schema.py        #    LoreChunkCreate, LoreSource
│   │   ├── chat_schema.py        #    ChatRequest, ChatResponse
│   │   └── health_schema.py      #    HealthResponse
│   ├── exceptions.py     #    Доменные исключения, не знающие про HTTP
│   ├── interfaces.py     #    Порты (Protocol): репозитории и AI-клиент — основа DIP
│   ├── repositories.py   # 3. Репозитории: BaseRepository, CharacterRepository, LoreChunkRepository
│   ├── services/         # 4-5. Сервисы, по файлу на ответственность
│   │   ├── ai_service.py             # 4. Внешний сервис: OpenAIClient (эмбеддинги + chat completions)
│   │   ├── rag_service.py            # 5. Бизнес-логика: RAGService
│   │   └── lore_ingestion_service.py # 5. Бизнес-логика: LoreIngestionService
│   ├── dependencies.py   # 6. DI-провайдеры для FastAPI Depends
│   ├── routers.py        # 6. Роутеры: GET /characters, POST /chat, GET /health
│   └── main.py           # 6. create_app(): lifespan, роутеры, обработчики исключений
├── tests/                # Юнит-тесты сервисов и интеграционные тесты API на фейках
├── seed.py               # 7. Инициализация схемы + демо-данные (идемпотентно)
├── docker-compose.yml    # PostgreSQL 16 + pgvector
└── requirements.txt
```

### Направление зависимостей

```
routers  ->  services  ->  interfaces (Protocol)  <-  repositories / services/ai_service
                    |
             schemas / models
```

* Пакеты `models/`, `schemas/` и `services/` держат по модулю на агрегат (или ответственность) и
  реэкспортируют публичные имена через `__init__.py` — импорты остаются стабильными
  (`from app.services import RAGService`), а файлы маленькими.
* `services/rag_service.py` и `services/lore_ingestion_service.py` не импортируют ни SQLAlchemy, ни
  OpenAI — только порты из `interfaces.py`.
* `repositories.py` — единственное место, где строится SQL.
* `services/ai_service.py` — единственное место, где используется SDK OpenAI.
* `routers.py` содержит только маршрутизацию, валидацию (через схемы) и ничего больше.
* Никакого глобального состояния: engine, сессия и AI-клиент живут в `app.state`, создаются в
  `lifespan` и раздаются через `Depends`.

### Принципы SOLID в коде

* **S** — каждый класс отвечает за одно: `LoreChunkRepository` только хранит и ищет чанки,
  `RAGService` только оркестрирует пайплайн, `OpenAIClient` только говорит с провайдером.
* **O** — новый тип внешнего сервиса (например, локальная модель) добавляется классом-адаптером,
  реализующим `AIClientProtocol`; существующий код не меняется.
* **L** — конкретные репозитории наследуют `BaseRepository[T]` и не сужают его контракт.
* **I** — порты разделены: сервису чата не нужен метод создания чанков, ингестору — метод поиска.
* **D** — `RAGService.__init__` принимает абстракции, а не `CharacterRepository(session)` и не
  `AsyncOpenAI`.

---

## Быстрый старт

### 1. Зависимости

Нужен Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # рантайм
pip install -r requirements-dev.txt    # + тесты, ruff, mypy
```

### 2. База данных

```bash
docker compose up -d                   # PostgreSQL 16 с расширением pgvector
```

### 3. Переменные окружения

```bash
cp .env.example .env
# впишите OPENAI_API_KEY
```

Все настройки читаются из окружения (ENV имеет приоритет над `.env`). Основные:

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://l4d:l4d@localhost:5432/l4d_terminal` | DSN для asyncpg |
| `OPENAI_API_KEY` | — | ключ OpenAI, без него приложение не стартует |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | модель эмбеддингов |
| `CHAT_MODEL` | `gpt-4o-mini` | модель ответов |
| `RAG_TOP_K` | `3` | сколько чанков подмешивать в контекст |
| `RAG_MIN_SIMILARITY` | `0.15` | порог косинусного сходства |
| `AUTO_CREATE_TABLES` | `false` | создать схему при старте вместо `seed.py` |
| `API_PREFIX` | `/api/v1` | префикс роутов; пустое значение отдаёт `/characters` и `/chat` |

### 4. Наполнение данными

```bash
python seed.py            # создаст расширение, таблицы, Эллиса, Билла и 6 чанков лора
python seed.py --reset    # сначала удалит таблицы
```

Скрипт идемпотентен: повторный запуск ничего не дублирует.

### 5. Запуск

```bash
uvicorn app.main:app --reload
```

* Swagger UI — http://127.0.0.1:8000/docs
* ReDoc — http://127.0.0.1:8000/redoc

---

## API

### `GET /api/v1/characters`

```bash
curl -s http://127.0.0.1:8000/api/v1/characters
```

```json
[
  {
    "id": 1,
    "name": "Ellis",
    "system_prompt": "You are Ellis, a cheerful young mechanic from Savannah...",
    "few_shot_examples": [{"user": "...", "assistant": "..."}]
  }
]
```

### `POST /api/v1/chat`

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"character_id": 1, "message": "How do we kill a Tank?"}'
```

```json
{
  "character_id": 1,
  "character_name": "Ellis",
  "answer": "Aw man, a Tank? Get your back to a wall and set the big guy on fire...",
  "model": "gpt-4o-mini",
  "sources": [
    {"id": 1, "data_type": "monster", "content": "The Tank is the rarest Special Infected...", "similarity": 0.83}
  ]
}
```

Поле `sources` возвращает чанки, на которых построен ответ, вместе с косинусным сходством — это
делает RAG наблюдаемым и упрощает отладку.

| Код | Когда |
|---|---|
| `200` | ответ сформирован |
| `404` | персонаж с таким `character_id` не найден |
| `422` | `message` пустой, длиннее 2000 символов или `character_id <= 0` |
| `502` | OpenAI вернул ошибку или не сконфигурирован ключ |

### `GET /api/v1/health`

Проверяет живость процесса и доступность PostgreSQL (`SELECT 1`), возвращая
`{"status": "ok", "version": "0.1.0", "database": "ok"}`.

---

## Как работает RAG-пайплайн

`RAGService.process_chat_message(character_id, user_message)`:

1. **Персона** — `CharacterRepository.get_by_id()`; если персонажа нет, сразу
   `CharacterNotFoundError` → `404`, без обращения к провайдеру.
2. **Эмбеддинг вопроса** — `OpenAIClient.get_embedding(user_message)` → вектор на 1536 чисел.
3. **Векторный поиск** — `LoreChunkRepository.search_similar_chunks(vector, limit=top_k)`
   выполняет в SQL `embedding <=> :query` (косинусное расстояние), сортирует по возрастанию
   расстояния и возвращает `similarity = 1 - distance`; индекс HNSW ускоряет поиск.
4. **Фильтрация** — чанки ниже `RAG_MIN_SIMILARITY` отбрасываются, чтобы нечёткий запрос не
   засорял промпт.
5. **Сборка промпта** — системный промпт персоны + блок `archive_records` с найденными фактами +
   few-shot примеры из `characters.few_shot_examples`.
6. **Генерация** — `OpenAIClient.generate_response(...)`, ответ упаковывается в `ChatResponse`
   вместе с источниками.

---

## Тесты

```bash
pytest                                  # всё: юнит-тесты + интеграционные, если поднят PostgreSQL
pytest tests/test_rag_service.py        # только бизнес-логика, без БД и сети
```

**Юнит-тесты** (`tests/test_rag_service.py`) не обращаются ни к PostgreSQL, ни к сети:
`tests/fakes.py` содержит in-memory реализации портов и детерминированный AI-клиент, а в
`tests/test_api.py` через `dependency_overrides` подменяются только внешние адаптеры — реальные
роутеры, DI-граф и `RAGService` при этом работают по-настоящему. Это и есть проверка того, что
инверсия зависимостей сделана корректно.

**Интеграционные тесты** (`tests/test_integration_postgres.py`) поднимают всё по-настоящему: свой
PostgreSQL-инстанс, реальные репозитории, косинусный поиск через pgvector и HNSW-индекс. OpenAI
подменяется локальным HTTP-заглушкой, поэтому ключ и интернет не нужны. Тест сам создаёт базу
`<имя_базы>_test`, а если PostgreSQL недоступен — скипается, оставляя прогон зелёным:

```bash
docker compose up -d
pytest tests/test_integration_postgres.py
```

```bash
ruff check .       # линтер
mypy               # статическая типизация (конфиг в pyproject.toml)
```

---

## Замечания по реализации

* `Base` — это `class Base(DeclarativeBase)`, а не `declarative_base()`: в SQLAlchemy 2.0 это
  каноничная форма, дающая нормальную типизацию `Mapped[...]` в подклассах.
* Сессии открываются на один HTTP-запрос (`Database.session` коммитит при успехе и откатывает при
  исключении), `expire_on_commit=False` позволяет безопасно отдавать ORM-объекты в `response_model`
  уже после закрытия сессии.
* Размерность вектора берётся из настроек (`EMBEDDING_DIMENSIONS`), а не хардкодится в модели,
  но должна совпадать с размерностью выбранной модели эмбеддингов.
* Комментарии в коде только там, где решение неочевидно (например, почему `<=>` — это расстояние,
  а не сходство).
