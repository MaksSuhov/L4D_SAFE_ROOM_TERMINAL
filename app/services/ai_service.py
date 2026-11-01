from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI, OpenAIError

from app.config import Settings
from app.exceptions import AIServiceError
from app.interfaces import AIClientProtocol

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger("l4d.ai")

NO_EVIDENCE_PLACEHOLDER: str = (
    "Архив не вернул ни одной подходящей записи по этому вопросу."
)

TRANSLATION_SYSTEM_PROMPT: str = (
    "Ты — переводчик игровой документации. Тебе присылают JSON-массив строк на "
    "английском языке. Переведи каждую строку на русский язык, сохранив смысл, "
    "структуру строк и оформление (стрелки «↳», символы «—», переносы строк). "
    "Имена собственные и названия из Left 4 Dead передавай так, как принято в "
    "русской локализации игры (Bill — Билл, Zoey — Зои, Ellis — Эллис, Tank — "
    "Танк, Witch — Ведьма, Boomer — Бумер, Smoker — Курильщик, Hunter — Охотник, "
    "Charger — Громила, Spitter — Плевун, Jockey — Жокей, Green Flu — Зелёный "
    "грипп, safe room — убежище). Отвечай ТОЛЬКО JSON-массивом той же длины, без "
    "пояснений и без markdown."
)


class OpenAIClient(AIClientProtocol):
    """Asynchronous wrapper around embeddings and chat completions."""

    def __init__(self, settings: Settings) -> None:
        if not settings.is_ai_configured:
            raise AIServiceError(
                "AI API key is missing: set AI_API_KEY (or ROUTERAI_API_KEY / "
                "OPENAI_API_KEY) in the environment or in .env"
            )

        self._settings = settings
        self.model_name: str = settings.chat_model
        self._client = AsyncOpenAI(
            api_key=settings.ai_api_key,
            base_url=settings.ai_base_url,
            timeout=settings.ai_timeout_seconds,
            max_retries=2,
        )

    async def get_embedding(self, text: str) -> list[float]:
        """Embed a single ``text`` with the configured embedding model."""
        vectors = await self.get_embeddings([text])
        return vectors[0]

    async def get_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` in batches, preserving their order.

        The corpus holds hundreds of records, so the texts travel in batches
        bounded by ``ingestion_batch_size``: one request per record would make
        seeding needlessly slow and would trip the provider's rate limits.
        """
        pending = list(texts)
        if not pending:
            return []

        vectors: list[list[float]] = []
        batch_size = self._settings.ingestion_batch_size
        for start in range(0, len(pending), batch_size):
            vectors.extend(await self._embed_batch(pending[start : start + batch_size]))
        return vectors

    async def translate_texts(self, texts: Sequence[str]) -> list[str]:
        """Translate ``texts`` into the configured corpus language, preserving order.

        The corpus is normalised once, while it is built, so the vector store and
        the questions share one language. A batch that comes back malformed is
        retried one text at a time, and a text that still fails keeps its
        original wording — a partially translated corpus is far better than a
        build that dies halfway through.
        """
        pending = list(texts)
        if not pending:
            return []

        translations: list[str] = []
        batch_size = self._settings.translation_batch_size
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            try:
                translations.extend(await self._translate_batch(batch))
            except AIServiceError as error:
                logger.warning("Batch translation failed, retrying one by one: %s", error)
                translations.extend(await self._translate_individually(batch))
        return translations

    async def generate_response(
        self,
        system_prompt: str,
        context: str,
        user_message: str,
        few_shot_examples: Sequence[Mapping[str, str]] | None = None,
    ) -> str:
        """Generate an in-character answer grounded in ``context``."""
        return await self._create_completion(
            self._compose_messages(system_prompt, context, user_message, few_shot_examples),
            temperature=self._settings.chat_temperature,
        )

    async def stream_response(
        self,
        system_prompt: str,
        context: str,
        user_message: str,
        few_shot_examples: Sequence[Mapping[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """Yield the same answer as :meth:`generate_response`, fragment by fragment."""
        messages = self._compose_messages(
            system_prompt, context, user_message, few_shot_examples
        )
        try:
            stream = await self._client.chat.completions.create(
                model=self._settings.chat_model,
                messages=messages,
                temperature=self._settings.chat_temperature,
                max_tokens=self._settings.chat_max_tokens,
                stream=True,
            )
        except OpenAIError as error:
            raise AIServiceError(f"Chat completion request failed: {error}") from error

        async for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.0,
    ) -> str:
        """Answer one instruction, without retrieval and without a persona."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return await self._create_completion(messages, temperature=temperature)

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.close()

    async def _create_completion(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        temperature: float,
    ) -> str:
        """Send one chat request and return its single text answer."""
        try:
            completion = await self._client.chat.completions.create(
                model=self._settings.chat_model,
                messages=list(messages),
                temperature=temperature,
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

    def _compose_messages(
        self,
        system_prompt: str,
        context: str,
        user_message: str,
        few_shot_examples: Sequence[Mapping[str, str]] | None,
    ) -> list[ChatCompletionMessageParam]:
        """Build the message list shared by the buffered and the streaming paths."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self._compose_system_message(system_prompt)},
            {"role": "system", "content": self._compose_context_message(context)},
        ]
        messages.extend(self._compose_examples(few_shot_examples))
        messages.append({"role": "user", "content": user_message})
        return messages

    async def _embed_batch(self, batch: Sequence[str]) -> list[list[float]]:
        """Embed one batch and return its vectors in the order of ``batch``."""
        request: dict[str, Any] = {
            "model": self._settings.embedding_model,
            "input": list(batch),
        }
        # Only the text-embedding-3 family accepts an explicit output width. The
        # test is a substring match because gateways prefix the id with the vendor
        # (openai/text-embedding-3-small).
        if "text-embedding-3" in self._settings.embedding_model:
            request["dimensions"] = self._settings.embedding_dimensions

        try:
            response = await self._client.embeddings.create(**request)
        except OpenAIError as error:
            raise AIServiceError(f"Embedding request failed: {error}") from error

        if len(response.data) != len(batch):
            raise AIServiceError(
                f"Embedding provider returned {len(response.data)} vectors "
                f"for a batch of {len(batch)} texts"
            )

        # The API is not required to return the items in the order they were
        # sent, so the index decides where each vector belongs.
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors: list[list[float]] = []
        for item in ordered:
            vector = [float(value) for value in item.embedding]
            if len(vector) != self._settings.embedding_dimensions:
                raise AIServiceError(
                    f"Embedding model returned {len(vector)} dimensions, "
                    f"but {self._settings.embedding_dimensions} were expected"
                )
            vectors.append(vector)
        return vectors

    async def _translate_batch(self, batch: Sequence[str]) -> list[str]:
        """Translate one batch, requiring exactly one translation per input."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(list(batch), ensure_ascii=False)},
        ]
        try:
            completion = await self._client.chat.completions.create(
                model=self._settings.chat_model,
                messages=messages,
                # Translation is a reconstruction task, not a creative one.
                temperature=0.0,
                max_tokens=self._settings.chat_max_tokens * 2,
            )
        except OpenAIError as error:
            raise AIServiceError(f"Translation request failed: {error}") from error

        if not completion.choices:
            raise AIServiceError("Translation response did not contain any choice")
        content = completion.choices[0].message.content or ""
        translated = self._extract_json_array(content)
        if len(translated) != len(batch):
            raise AIServiceError(
                f"Translation returned {len(translated)} items for a batch of {len(batch)}"
            )
        return translated

    async def _translate_individually(self, batch: Sequence[str]) -> list[str]:
        """Translate a batch one text at a time, keeping the original on failure."""
        translations: list[str] = []
        for text in batch:
            try:
                translations.extend(await self._translate_batch([text]))
            except AIServiceError as error:
                logger.warning("Keeping the original text, translation failed: %s", error)
                translations.append(text)
        return translations

    @staticmethod
    def _extract_json_array(content: str) -> list[str]:
        """Pull a JSON string array out of a model reply, tolerating prose around it."""
        start = content.find("[")
        end = content.rfind("]")
        if start == -1 or end == -1 or end < start:
            raise AIServiceError("Translation reply did not contain a JSON array")
        try:
            parsed = json.loads(content[start : end + 1])
        except json.JSONDecodeError as error:
            raise AIServiceError(f"Translation reply was not valid JSON: {error}") from error
        if not isinstance(parsed, list):
            raise AIServiceError("Translation reply was not a JSON array")
        return [
            item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            for item in parsed
        ]

    def _compose_system_message(self, system_prompt: str) -> str:
        """Append the configured language instruction to the persona prompt."""
        instruction = self._settings.answer_language_instruction.strip()
        if not instruction:
            return system_prompt
        return f"{system_prompt}\n\n{instruction}"

    @staticmethod
    def _compose_context_message(context: str) -> str:
        """Wrap the evidence message, substituting a placeholder when empty.

        The message is assembled by the prompt builder, which owns the wording
        of the two evidence tiers; the provider adapter only transports it.
        """
        return context.strip() or NO_EVIDENCE_PLACEHOLDER

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
