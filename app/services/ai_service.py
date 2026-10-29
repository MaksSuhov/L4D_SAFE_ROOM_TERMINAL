"""External AI service: a thin adapter over the OpenAI SDK.

Keeping the SDK behind :class:`AIClientProtocol` means the RAG pipeline can be
tested with a fake client and that swapping the provider touches only this file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI, OpenAIError

from app.config import Settings
from app.exceptions import AIServiceError
from app.interfaces import AIClientProtocol

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from openai.types.chat import ChatCompletionMessageParam

CONTEXT_INSTRUCTION: str = (
    "The records below come from the survivors' field archive. Treat them as the "
    "authoritative source of facts, prefer them over your general knowledge, and "
    "answer in character. If the records do not cover the question, say so instead "
    "of inventing game mechanics."
)

NO_CONTEXT_PLACEHOLDER: str = (
    "The archive returned no matching records for this question."
)


class OpenAIClient(AIClientProtocol):
    """Asynchronous wrapper around embeddings and chat completions."""

    def __init__(self, settings: Settings) -> None:
        if not settings.is_openai_configured:
            raise AIServiceError(
                "OpenAI API key is missing: set OPENAI_API_KEY in the environment or in .env"
            )

        self._settings = settings
        self.model_name: str = settings.chat_model
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.openai_timeout_seconds,
            max_retries=2,
        )

    async def get_embedding(self, text: str) -> list[float]:
        """Embed ``text`` with the configured embedding model."""
        request: dict[str, Any] = {
            "model": self._settings.embedding_model,
            "input": text,
        }
        # Only the text-embedding-3 family accepts an explicit output width.
        if self._settings.embedding_model.startswith("text-embedding-3"):
            request["dimensions"] = self._settings.embedding_dimensions

        try:
            response = await self._client.embeddings.create(**request)
        except OpenAIError as error:
            raise AIServiceError(f"Embedding request failed: {error}") from error

        if not response.data:
            raise AIServiceError("Embedding response did not contain any vector")

        vector = [float(value) for value in response.data[0].embedding]
        if len(vector) != self._settings.embedding_dimensions:
            raise AIServiceError(
                f"Embedding model returned {len(vector)} dimensions, "
                f"but {self._settings.embedding_dimensions} were expected"
            )
        return vector

    async def generate_response(
        self,
        system_prompt: str,
        context: str,
        user_message: str,
        few_shot_examples: Sequence[Mapping[str, str]] | None = None,
    ) -> str:
        """Generate an in-character answer grounded in ``context``."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": self._compose_context_message(context)},
        ]
        messages.extend(self._compose_examples(few_shot_examples))
        messages.append({"role": "user", "content": user_message})

        try:
            completion = await self._client.chat.completions.create(
                model=self._settings.chat_model,
                messages=messages,
                temperature=self._settings.chat_temperature,
                max_tokens=self._settings.chat_max_tokens,
            )
        except OpenAIError as error:
            raise AIServiceError(f"Chat completion request failed: {error}") from error

        if not completion.choices:
            raise AIServiceError("Chat completion response did not contain any choice")

        answer = completion.choices[0].message.content
        if not answer:
            raise AIServiceError("Chat completion returned an empty message")
        return answer.strip()

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.close()

    @staticmethod
    def _compose_context_message(context: str) -> str:
        body = context.strip() or NO_CONTEXT_PLACEHOLDER
        return f"{CONTEXT_INSTRUCTION}\n\n<archive_records>\n{body}\n</archive_records>"

    @staticmethod
    def _compose_examples(
        few_shot_examples: Sequence[Mapping[str, str]] | None,
    ) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        for example in few_shot_examples or ():
            user_turn = example.get("user")
            assistant_turn = example.get("assistant")
            if not user_turn or not assistant_turn:
                continue
            messages.append({"role": "user", "content": user_turn})
            messages.append({"role": "assistant", "content": assistant_turn})
        return messages
