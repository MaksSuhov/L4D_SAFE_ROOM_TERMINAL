"""Снимок корпуса на диске.

Загруженные и разобранные записи пишутся в файл JSONL, который коммитится в
репозиторий. Наполнение базы затем читает этот файл вместо вики, за счёт чего
векторное хранилище воспроизводимо, проверяемо по диффу и собирается без
доступа к сети. Сырая разметка, лежащая за снимком, остаётся в кэше викитекста,
так что снимок всегда можно пересобрать из его источника.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from app.exceptions import ApplicationError
from app.schemas import LoreChunkCreate


class CorpusSnapshot:
    """Читает и пишет снимок корпуса в JSONL."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """Расположение файла снимка."""
        return self._path

    def exists(self) -> bool:
        """Был ли снимок уже записан."""
        return self._path.is_file()

    def write(self, documents: Sequence[LoreChunkCreate]) -> None:
        """Записать ``documents`` по одному объекту JSON на строку, атомарно.

        Запись идёт через временный файл, чтобы прерванная сборка не оставила
        после себя усечённый корпус.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(f"{self._path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for document in documents:
                handle.write(document.model_dump_json())
                handle.write("\n")
        temporary.replace(self._path)

    def read(self) -> list[LoreChunkCreate]:
        """Прочитать каждую запись снимка, отвергая некорректные строки."""
        if not self.exists():
            raise ApplicationError(
                f"Corpus snapshot {self._path} is missing; build it with "
                "'python seed.py --refresh'"
            )

        documents: list[LoreChunkCreate] = []
        with self._path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    documents.append(LoreChunkCreate.model_validate(json.loads(stripped)))
                except (json.JSONDecodeError, ValueError) as error:
                    raise ApplicationError(
                        f"{self._path}: line {number} is not a valid corpus record: {error}"
                    ) from error
        return documents

    def summarise(self) -> dict[str, int]:
        """Посчитать сохранённые записи по уровням свидетельств."""
        return dict(Counter(str(document.source_type) for document in self.read()))
