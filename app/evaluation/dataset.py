"""Чтение оценок из JSONL-датасета.

Формат простой: одна строка — один JSON-объект. Пустые строки и строки,
начинающиеся с ``#``, пропускаются, поэтому в файле можно оставлять заметки.

Пример строки::

    {"question": "Кто такой Танк?", "character_id": 5,
     "reference_answer": "Мутировавший выживший...", "expected_sources": ["Танк"]}
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from app.evaluation.domain import EvalCase
from app.exceptions import EvaluationError


def load_cases(path: Path) -> list[EvalCase]:
    """Прочитать датасет целиком."""
    if not path.exists():
        raise EvaluationError(f"Dataset not found: {path}")

    records: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise EvaluationError(f"{path}:{number}: invalid JSON: {error}") from error
            if not isinstance(parsed, dict):
                raise EvaluationError(f"{path}:{number}: expected a JSON object")
            records.append(parsed)

    return [
        build_case(record, where=f"{path}:{number}")
        for number, record in enumerate(records, start=1)
    ]


def build_case(record: Mapping[str, Any], *, where: str) -> EvalCase:
    """Собрать кейс из записи, внятно объяснив, что именно не так."""
    question = str(record.get("question", "")).strip()
    if not question:
        raise EvaluationError(f"{where}: 'question' must not be empty")

    character_id = record.get("character_id")
    if not isinstance(character_id, int) or isinstance(character_id, bool) or character_id <= 0:
        raise EvaluationError(f"{where}: 'character_id' must be a positive integer")

    return EvalCase(
        question=question,
        character_id=character_id,
        reference_answer=str(record.get("reference_answer", "")).strip(),
        expected_sources=_as_text_tuple(record.get("expected_sources"), where, "expected_sources"),
        tags=_as_text_tuple(record.get("tags"), where, "tags"),
    )


def _as_text_tuple(value: object, where: str, field: str) -> tuple[str, ...]:
    """Превратить необязательный список строк в кортеж строк."""
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise EvaluationError(f"{where}: '{field}' must be a list of strings")
    return tuple(str(item).strip() for item in value if str(item).strip())
