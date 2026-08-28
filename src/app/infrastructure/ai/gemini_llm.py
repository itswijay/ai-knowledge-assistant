from typing import Protocol

from google import genai
from google.genai import errors, types

from app.domain.errors import LLMGenerationError

GEMINI_3_PREFIX = "gemini-3"
LEGACY_TEMPERATURE = 0.1


class AsyncModelsClient(Protocol):
    async def generate_content(
        self,
        *,
        model: str,
        contents: object,
        config: types.GenerateContentConfig,
    ) -> types.GenerateContentResponse: ...


class AsyncGeminiClient(Protocol):
    models: AsyncModelsClient

    async def aclose(self) -> None: ...


class GeminiClient(Protocol):
    aio: AsyncGeminiClient


class GeminiLLMProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_output_tokens: int,
        client: GeminiClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        if not model.strip():
            raise ValueError("model must not be blank")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")

        self._model = model
        self._max_output_tokens = max_output_tokens
        self._client = client or genai.Client(api_key=api_key)

    async def generate(self, *, system_instruction: str, content: str) -> str:
        if not system_instruction.strip():
            raise LLMGenerationError("System instruction must not be blank")
        if not content.strip():
            raise LLMGenerationError("Content must not be blank")

        config = self._generation_config(system_instruction)
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=content,
                config=config,
            )
        except errors.APIError as error:
            raise LLMGenerationError("Gemini answer generation failed") from error

        answer = response.text
        if answer is None or not answer.strip():
            raise LLMGenerationError("Gemini returned an empty answer")
        return answer.strip()

    async def close(self) -> None:
        await self._client.aio.aclose()

    def _generation_config(
        self,
        system_instruction: str,
    ) -> types.GenerateContentConfig:
        if self._model.startswith(GEMINI_3_PREFIX):
            return types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=self._max_output_tokens,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW
                ),
            )
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=self._max_output_tokens,
            temperature=LEGACY_TEMPERATURE,
        )
