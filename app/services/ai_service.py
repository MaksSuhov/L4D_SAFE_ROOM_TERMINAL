from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from langfuse.openai import AsyncOpenAI
from openai import OpenAIError

from app.config import Settings
from app.exceptions import AIServiceError
from app.interfaces import AIClientProtocol
from app.services.json_array import extract_json_array

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
    """Асинхронная обёртка над эмбеддингами и чат-запросами."""

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
        """Получить эмбеддинг одного ``text`` настроенной моделью эмбеддингов."""
        vectors = await self.get_embeddings([text])
        return vectors[0]

    async def get_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """Получить эмбеддинги для ``texts`` пакетами, сохраняя их порядок.

        В корпусе сотни записей, поэтому тексты идут пакетами, ограниченными
        ``ingestion_batch_size``: по одному запросу на запись наполнение базы шло бы
        без нужды медленно и упёрлось бы в лимиты частоты провайдера.
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
        """Перевести ``texts`` на настроенный язык корпуса, сохранив порядок.

        Корпус нормализуется один раз, при сборке, чтобы векторное хранилище и
        вопросы говорили на одном языке. Пакет, который вернулся испорченным,
        переводится заново по одному тексту, а текст, который всё равно не поддался,
        остаётся в исходной формулировке: частично переведённый корпус много лучше
        сборки, умершей на полпути.
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
        """Сгенерировать ответ в образе, опираясь на ``context``."""
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
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Отдавать тот же ответ, что :meth:`generate_response`, фрагмент за фрагментом.

        ``temperature`` переопределяет настройку на один вызов: стенд оценки
        закрепляет её, чтобы прогоны были сравнимы, а боевой чат оставляет значение
        по умолчанию.

        Поток без единого фрагмента — это ошибка, а не пустой ответ: так провайдер
        отвечает при сбое на своей стороне, и молча вернуть пустоту значило бы не
        отдать игроку ничего, а стенду — посчитать это ответом. Проверка повторяет
        ту, что уже стоит в :meth:`_create_completion`.
        """
        messages = self._compose_messages(
            system_prompt, context, user_message, few_shot_examples
        )
        try:
            stream = await self._client.chat.completions.create(
                model=self._settings.chat_model,
                messages=messages,
                temperature=(
                    self._settings.chat_temperature if temperature is None else temperature
                ),
                max_tokens=self._settings.chat_max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                stream=True,
            )
        except OpenAIError as error:
            raise AIServiceError(f"Chat completion request failed: {error}") from error

        produced = False
        async for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                produced = True
                yield content

        if not produced:
            raise AIServiceError("Chat completion stream returned no content")

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.0,
    ) -> str:
        """Ответить на одну инструкцию, без поиска и без образа."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return await self._create_completion(messages, temperature=temperature)

    async def aclose(self) -> None:
        """Освободить нижележащий пул HTTP-соединений."""
        await self._client.close()

    async def _create_completion(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        temperature: float,
    ) -> str:
        """Отправить один чат-запрос и вернуть единственный текстовый ответ."""
        try:
            completion = await self._client.chat.completions.create(
                model=self._settings.chat_model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=self._settings.chat_max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
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
        """Собрать список сообщений, общий для буферизованного и потокового путей."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self._compose_system_message(system_prompt)},
            {"role": "system", "content": self._compose_context_message(context)},
        ]
        messages.extend(self._compose_examples(few_shot_examples))
        messages.append({"role": "user", "content": user_message})
        return messages

    async def _embed_batch(self, batch: Sequence[str]) -> list[list[float]]:
        """Получить эмбеддинги одного пакета в порядке ``batch``."""
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
        """Перевести один пакет, требуя ровно один перевод на каждый вход."""
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
        translated = extract_json_array(content)
        if len(translated) != len(batch):
            raise AIServiceError(
                f"Translation returned {len(translated)} items for a batch of {len(batch)}"
            )
        return translated

    async def _translate_individually(self, batch: Sequence[str]) -> list[str]:
        """Перевести пакет по одному тексту, оставляя исходный при неудаче."""
        translations: list[str] = []
        for text in batch:
            try:
                translations.extend(await self._translate_batch([text]))
            except AIServiceError as error:
                logger.warning("Keeping the original text, translation failed: %s", error)
                translations.append(text)
        return translations

    def _compose_system_message(self, system_prompt: str) -> str:
        """Дописать настроенную языковую инструкцию к промпту образа."""
        instruction = self._settings.answer_language_instruction.strip()
        if not instruction:
            return system_prompt
        return f"{system_prompt}\n\n{instruction}"

    @staticmethod
    def _compose_context_message(context: str) -> str:
        """Обернуть сообщение с доказательствами, подставив заглушку, когда оно пусто.

        Сообщение собирает конструктор промптов, которому принадлежат формулировки
        двух уровней доказательств; адаптер провайдера только передаёт его.
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
