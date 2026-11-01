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
from app.dependencies import get_rag_service
from app.evaluation import CaseEvaluator, EvaluationReport, EvaluationService, LLMJudge, load_cases
from app.exceptions import AIServiceError, EvaluationError
from app.repositories import CharacterRepository, LoreChunkRepository
from app.services import ArchivePromptBuilder, OpenAIClient

logger = logging.getLogger("l4d.eval")


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Evaluate the terminal with an LLM judge.")
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


async def evaluate(*, dataset: Path, limit: int) -> EvaluationReport:
    """Прогнать датасет через реальный пайплайн и вернуть отчёт."""
    settings: Settings = get_settings()

    cases = load_cases(dataset)
    if limit > 0:
        cases = cases[:limit]
    if not cases:
        raise EvaluationError(f"Dataset is empty: {dataset}")

    client = OpenAIClient(settings)
    database = Database(settings)
    try:
        async with database.session() as session:
            rag_service = get_rag_service(
                CharacterRepository(session),
                LoreChunkRepository(session),
                ArchivePromptBuilder(),
                client,
                settings,
            )
            evaluator = CaseEvaluator(
                context_builder=rag_service,
                model=client,
                judge=LLMJudge(client),
            )
            return await EvaluationService(evaluator).run(cases)
    finally:
        await database.dispose()
        await client.aclose()


def write_report(path: Path, report: EvaluationReport) -> None:
    """Записать отчёт в JSON-файл, создав каталог при необходимости."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(arguments: argparse.Namespace) -> int:
    """Синхронная обёртка: прогнать, напечатать отчёт, записать JSON."""
    try:
        report = asyncio.run(evaluate(dataset=arguments.dataset, limit=arguments.limit))
    except (EvaluationError, AIServiceError) as error:
        logger.error("Evaluation failed: %s", error)
        return 1

    print(report.render_text())
    if arguments.output is not None:
        write_report(arguments.output, report)
        logger.info("Report written to %s", arguments.output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    sys.exit(run(parse_args()))
