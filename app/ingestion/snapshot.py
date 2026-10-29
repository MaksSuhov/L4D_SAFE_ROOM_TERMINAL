"""The on-disk corpus snapshot.

Fetched and parsed records are written to a JSONL file that is committed to the
repository. Seeding then reads that file instead of the wiki, which makes the
vector store reproducible, reviewable in a diff and buildable without network
access. The raw markup behind a snapshot stays in the wikitext cache, so the
snapshot can always be regenerated from its source.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from app.exceptions import ApplicationError
from app.schemas import LoreChunkCreate


class CorpusSnapshot:
    """Reads and writes the JSONL corpus snapshot."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """Location of the snapshot file."""
        return self._path

    def exists(self) -> bool:
        """Whether a snapshot has been written already."""
        return self._path.is_file()

    def write(self, documents: Sequence[LoreChunkCreate]) -> None:
        """Write ``documents`` as one JSON object per line, atomically.

        The write goes through a temporary file so an interrupted build cannot
        leave a truncated corpus behind.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(f"{self._path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for document in documents:
                handle.write(document.model_dump_json())
                handle.write("\n")
        temporary.replace(self._path)

    def read(self) -> list[LoreChunkCreate]:
        """Read every record of the snapshot, rejecting malformed lines."""
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
        """Count the stored records per evidence tier."""
        return dict(Counter(str(document.source_type) for document in self.read()))
