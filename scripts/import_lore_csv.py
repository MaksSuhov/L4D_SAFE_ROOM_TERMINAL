"""Перенос архива лора из выгрузки pgvector в Qdrant.

Postgres хранит не только текст, но и уже оплаченные эмбеддинги, поэтому перелить
их дешевле, чем считать заново: размерность и метрика совпадают, платить
провайдеру не за что. Скрипт читает плоскую выгрузку таблицы и записывает точки
Qdrant, досчитывая только разреженный вектор BM25, которого в Postgres не было.

Первый шаг делается до вывода контейнера базы::

    docker exec l4d_terminal_db psql -U l4d -d l4d_terminal -c "\\copy (
        SELECT id, content, content_hash, source_type, topic, title,
               character_name, source_url, embedding::text
        FROM lore_chunks ORDER BY id
    ) TO STDOUT WITH CSV HEADER" > data/migration/lore_chunks.csv

Затем::

    .venv/bin/python -m scripts.import_lore_csv
    .venv/bin/python -m scripts.import_lore_csv --csv data/migration/lore_chunks.csv

Колонка ``id`` выгрузки не используется: идентификатор точки выводится из хеша
текста, чтобы повторный перенос перезаписывал те же записи, а не удваивал архив.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from app.config import Settings, get_settings
from app.dependencies import build_sparse_encoder
from app.exceptions import ApplicationError
from app.schemas import EmbeddedLoreChunk
from app.taxonomy import LoreTopic, SourceType
from app.vector import QdrantLoreChunkRepository, VectorStore

logger = logging.getLogger("l4d.import")

DEFAULT_CSV: Path = Path("data/migration/lore_chunks.csv")
DEFAULT_BATCH_SIZE: int = 256


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Перенести лор из pgvector в Qdrant.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="выгрузка таблицы лора")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="записей за раз")
    return parser.parse_args()


def read_records(path: Path, *, dimensions: int) -> Iterator[EmbeddedLoreChunk]:
    """Прочитать выгрузку и превратить каждую строку в запись корпуса.

    Размерность проверяется сразу: коллекция создаётся под модель из настроек, и
    без этой проверки несовпадение вылезло бы отказом Qdrant на середине заливки,
    ничего не говоря о причине.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            embedding = json.loads(row["embedding"])
            if len(embedding) != dimensions:
                raise ApplicationError(
                    f"Embedding of {row['content_hash'][:12]}… has {len(embedding)} dimensions, "
                    f"but the collection is built for {dimensions}; "
                    "check EMBEDDING_MODEL and EMBEDDING_DIMENSIONS"
                )
            yield EmbeddedLoreChunk(
                content=row["content"],
                source_type=SourceType(row["source_type"]),
                topic=LoreTopic(row["topic"]),
                title=row["title"],
                character_name=(row.get("character_name") or "").strip() or None,
                source_url=row.get("source_url") or "",
                content_hash=row["content_hash"],
                embedding=embedding,
            )


def batches(records: Iterable[EmbeddedLoreChunk], size: int) -> Iterator[list[EmbeddedLoreChunk]]:
    """Нарезать поток записей на пачки: корпус целиком в память не поднимается."""
    window: list[EmbeddedLoreChunk] = []
    for record in records:
        window.append(record)
        if len(window) >= size:
            yield window
            window = []
    if window:
        yield window


async def import_records(settings: Settings, path: Path, batch_size: int) -> int:
    """Залить выгрузку в коллекцию архива и вернуть число перенесённых записей."""
    vector_store = VectorStore(settings)
    try:
        await vector_store.ensure_collections()
        repository = QdrantLoreChunkRepository(
            vector_store.client,
            vector_store.lore_collection,
            build_sparse_encoder(settings),
            hybrid_enabled=settings.rag_hybrid_enabled,
            prefetch_multiplier=settings.rag_prefetch_multiplier,
        )

        total = 0
        records = read_records(path, dimensions=settings.embedding_dimensions)
        for batch in batches(records, batch_size):
            await repository.create_chunks(batch)
            total += len(batch)
            logger.info("Imported %d record(s)", total)
        return total
    finally:
        await vector_store.close()


def main() -> None:
    """Точка входа скрипта."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    arguments = parse_args()
    if not arguments.csv.exists():
        raise SystemExit(f"CSV not found: {arguments.csv}")

    settings = get_settings()
    total = asyncio.run(import_records(settings, arguments.csv, arguments.batch_size))
    print(f"Imported {total} record(s) into {settings.lore_collection}")


if __name__ == "__main__":
    main()
