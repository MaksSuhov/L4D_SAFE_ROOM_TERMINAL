"""Normalising the corpus language before it is embedded.

The wiki is written in English while the terminal answers in Russian, and
cross-language vector search is weaker than monolingual search: in a spot check
against this very corpus a Russian question about the Green Flu scored 0.53
against the English record but 0.74 against its Russian counterpart, with a much
wider margin over the runner-up. Retrieval still ranked the right record first in
every case, so the corpus is usable as it stands — but a deployment that wants
the tighter margins can normalise the corpus once, here.

The stage is **opt-in** (``--translate``), deliberately. It costs one chat
completion per batch of records, and over the full corpus that is a slow pass:
measured against the configured provider, sixteen short lines took ~21 s, and a
single long article paragraph took ~47 s and came back untranslated. Over
~2 300 records that is a tens-of-minutes job — reasonable as a one-off for a
corpus that will be shipped, wrong as a hidden default for every local seed. A
text the provider fails to translate keeps its source form rather than being
dropped, so a partial pass degrades instead of corrupting the archive.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.interfaces import AIClientProtocol
from app.schemas import LoreChunkCreate

logger = logging.getLogger("l4d.ingestion.translation")


@dataclass(frozen=True, slots=True)
class TranslationOutcome:
    """Result of translating a corpus."""

    documents: list[LoreChunkCreate]
    translated: int
    unchanged: int

    @property
    def total(self) -> int:
        """Number of records that went in."""
        return self.translated + self.unchanged


class CorpusTranslator:
    """Rewrites the ``content`` of every record in the target language."""

    def __init__(self, ai_client: AIClientProtocol, *, target_language: str) -> None:
        self._ai_client = ai_client
        self._target_language = target_language

    @property
    def target_language(self) -> str:
        """Language every record is normalised into."""
        return self._target_language

    async def translate(self, documents: Sequence[LoreChunkCreate]) -> TranslationOutcome:
        """Translate the records and report how many were actually rewritten."""
        if not documents:
            return TranslationOutcome(documents=[], translated=0, unchanged=0)

        originals = [document.content for document in documents]
        logger.info(
            "Translating %d corpus record(s) into %s",
            len(originals),
            self._target_language,
        )
        translations = await self._ai_client.translate_texts(originals)

        translated_documents: list[LoreChunkCreate] = []
        translated = 0
        for document, original, translation in zip(documents, originals, translations, strict=True):
            replacement = translation.strip()
            # An empty reply would silently erase a record; the source text is
            # always a valid fallback.
            if not replacement:
                replacement = original
            if replacement != original:
                translated += 1
            translated_documents.append(
                document.model_copy(update={"content": replacement})
            )
        return TranslationOutcome(
            documents=translated_documents,
            translated=translated,
            unchanged=len(documents) - translated,
        )
