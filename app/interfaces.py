"""Ports of the application (Dependency Inversion Principle).

The service layer depends on these ``Protocol`` definitions rather than on the
concrete SQLAlchemy repositories or on the OpenAI SDK wrapper. Any object with a
matching shape — a real adapter, a fake used in tests, or a future vector
database client — can be injected instead.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

from app.models import Character, LoreChunk
from app.schemas import EmbeddedLoreChunk, LoreSource
from app.taxonomy import SourceType


class RetrievedChunk(NamedTuple):
    """A lore chunk paired with the cosine similarity of a search query."""

    chunk: LoreChunk
    similarity: float


@dataclass(frozen=True, slots=True)
class PreparedChat:
    """Everything a single chat turn needs once retrieval has run.

    It is the seam between the two halves of the pipeline: retrieval fills it in
    (question -> evidence -> brief), and generation only needs what is stored
    here. Exposing it lets a caller stream the answer and keep the evidence,
    which is exactly what the offline evaluation harness does.
    """

    character_id: int
    character_name: str
    model: str
    system_prompt: str
    context: str
    few_shot_examples: Sequence[Mapping[str, str]]
    sources: Sequence[LoreSource]


class CharacterRepositoryProtocol(Protocol):
    """Read access to character personas."""

    async def get_by_id(self, entity_id: int) -> Character | None:
        """Return the character with the given identifier, or ``None``."""
        ...

    async def get_all(self) -> Sequence[Character]:
        """Return every stored character ordered by identifier."""
        ...


class LoreChunkRepositoryProtocol(Protocol):
    """Persistence and vector search over lore fragments."""

    async def create_chunk(self, record: EmbeddedLoreChunk) -> LoreChunk:
        """Persist a single embedded lore fragment."""
        ...

    async def create_chunks(self, records: Sequence[EmbeddedLoreChunk]) -> Sequence[LoreChunk]:
        """Persist several embedded lore fragments in one round trip."""
        ...

    async def search_similar_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        source_types: Sequence[SourceType] | None = None,
        topic: str | None = None,
        character_name: str | None = None,
    ) -> Sequence[RetrievedChunk]:
        """Return the ``limit`` closest chunks ordered by descending similarity.

        ``source_types`` restricts the search to one or more evidence tiers,
        which is what makes the two-level retrieval possible: the facts are
        looked up among the official records while the rumours are looked up
        among the graffiti.
        """
        ...

    async def existing_content_hashes(self, hashes: Sequence[str]) -> set[str]:
        """Return which of the given content hashes are already stored."""
        ...

    async def count(self) -> int:
        """Return the number of stored lore fragments."""
        ...


class AIClientProtocol(Protocol):
    """Access to the language model provider."""

    model_name: str
    """Name of the chat model, surfaced in API responses."""

    async def get_embedding(self, text: str) -> list[float]:
        """Embed ``text`` into a vector of ``embedding_dimensions`` floats."""
        ...

    async def get_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, preserving their order.

        Batching matters: the corpus holds hundreds of records, and one request
        per record would turn seeding into a coffee break.
        """
        ...

    async def translate_texts(self, texts: Sequence[str]) -> list[str]:
        """Translate a batch of texts into the configured corpus language.

        Used once, while the corpus is built, so an English wiki becomes a
        corpus in the language the terminal answers in. Implementations must
        return exactly one translation per input, falling back to the original
        text rather than dropping a record.
        """
        ...

    async def generate_response(
        self,
        system_prompt: str,
        context: str,
        user_message: str,
        few_shot_examples: Sequence[Mapping[str, str]] | None = None,
    ) -> str:
        """Produce the assistant answer grounded in ``context``."""
        ...

    def stream_response(
        self,
        system_prompt: str,
        context: str,
        user_message: str,
        few_shot_examples: Sequence[Mapping[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """Yield the answer piece by piece, in the order the provider sends it.

        The same answer as :meth:`generate_response`, only delivered as a stream
        of fragments. Streaming is what makes the time to the first token
        observable: until the first fragment arrives, nothing has been produced.
        """
        ...

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.0,
    ) -> str:
        """Answer a single instruction, without retrieval and without personas.

        A low-level entry point for tasks that only need a model reply: judging
        an answer, classifying a text, extracting a field. The temperature is an
        argument here because such tasks want determinism, not creativity.
        """
        ...
