from __future__ import annotations

import json

from app.exceptions import AIServiceError


def extract_json_array(content: str) -> list[str]:
    """Вытащить JSON-массив строк из ответа модели, выдержав текст вокруг него."""
    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise AIServiceError("Model reply did not contain a JSON array")
    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError as error:
        raise AIServiceError(f"Model reply was not valid JSON: {error}") from error
    if not isinstance(parsed, list):
        raise AIServiceError("Model reply was not a JSON array")
    return [
        item if isinstance(item, str) else json.dumps(item, ensure_ascii=False) for item in parsed
    ]
