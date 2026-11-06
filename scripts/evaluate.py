"""Точка входа: прогнать датасет через терминал и оценить ответы.

Запуск из корня проекта (нужны поднятая база, наполненный архив и AI_API_KEY)::

    make eval
    .venv/bin/python -m scripts.evaluate --dataset data/eval/questions.jsonl --limit 5

Скрипт собирает те же зависимости, что и HTTP-слой (``get_rag_service``), поэтому
оценивается ровно тот пайплайн, который отвечает игрокам, а не его копия.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from app.config import Settings, get_settings
from app.database import Database
from app.dependencies import build_query_expander, get_rag_service
from app.evaluation import (
    CaseEvaluator,
    EmbeddingSimilarity,
    EvaluationReport,
    EvaluationService,
    LLMJudge,
    load_cases,
)
from app.exceptions import AIServiceError, EvaluationError
from app.repositories import CharacterRepository, LoreChunkRepository
from app.services import ArchivePromptBuilder, OpenAIClient, build_tracer

logger = logging.getLogger("l4d.eval")


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Оценить терминал с помощью LLM-судьи.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/questions.jsonl"),
        help="датасет с вопросами (JSONL, по объекту на строку)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/report.json"),
        help="куда записать отчёт в JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="оценить только первые N кейсов; 0 — все",
    )
    return parser.parse_args()


async def evaluate(*, dataset: Path, limit: int) -> tuple[EvaluationReport, dict[str, object]]:
    """Прогнать датасет через реальный пайплайн и вернуть отчёт с настройками прогона."""
    settings: Settings = get_settings()

    cases = load_cases(dataset)
    if limit > 0:
        cases = cases[:limit]
    if not cases:
        raise EvaluationError(f"Dataset is empty: {dataset}")

    client = OpenAIClient(settings)
    database = Database(settings)
    tracer = build_tracer(settings)
    try:
        async with database.session() as session:
            rag_service = get_rag_service(
                CharacterRepository(session),
                LoreChunkRepository(session),
                ArchivePromptBuilder(),
                client,
                tracer,
                settings,
                build_query_expander(client, settings, tracer),
            )
            evaluator = CaseEvaluator(
                context_builder=rag_service,
                model=client,
                judge=LLMJudge(client),
                similarity=EmbeddingSimilarity(client),
                attempts=settings.eval_attempts,
                temperature=settings.eval_temperature,
            )
            report = await EvaluationService(evaluator).run(cases)
    finally:
        # The process exits as soon as this returns, so the tracer has to be drained
        # here: batching would otherwise drop the whole run on the floor.
        tracer.flush()
        tracer.shutdown()
        await database.dispose()
        await client.aclose()

    config: dict[str, object] = {
        "dataset": str(dataset),
        "model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "temperature": settings.eval_temperature,
        "attempts": settings.eval_attempts,
        "limit": limit,
    }
    return report, config


def write_report(path: Path, report: EvaluationReport, config: dict[str, object]) -> None:
    """Записать отчёт в JSON-файл, создав каталог при необходимости.

    Настройки прогона кладутся рядом с числами: без них отчёт невозможно
    сравнить с прошлым — непонятно, отвечала та же модель и с той же температурой
    или числа успели разъехаться из-за конфигурации.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": config, **report.to_dict()}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(arguments: argparse.Namespace) -> int:
    """Синхронная обёртка: прогнать, напечатать отчёт, записать JSON."""
    try:
        report, config = asyncio.run(
            evaluate(dataset=arguments.dataset, limit=arguments.limit)
        )
    except (EvaluationError, AIServiceError) as error:
        logger.error("Evaluation failed: %s", error)
        return 1

    print(report.render_text())
    if arguments.output is not None:
        write_report(arguments.output, report, config)
        logger.info("Report written to %s", arguments.output)

    # Ненулевой код возврата при сбоях: молча отдать неполный отчёт как успешный
    # прогон — это ровно та ошибка, из-за которой сбойный кейс раньше выглядел
    # отличным ответом.
    if report.failed_count:
        logger.error("%d case(s) produced no answer; see the report", report.failed_count)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    sys.exit(run(parse_args()))
