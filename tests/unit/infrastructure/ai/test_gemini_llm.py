from dataclasses import dataclass, field

import pytest
from google.genai import errors, types

from app.domain.errors import LLMGenerationError
from app.infrastructure.ai import GeminiLLMProvider


@dataclass
class FakeModelsClient:
    response: types.GenerateContentResponse
    error: errors.APIError | None = None
    calls: list[dict[str, object]] = field(default_factory=list)

    async def generate_content(
        self,
        *,
        model: str,
        contents: object,
        config: types.GenerateContentConfig,
    ) -> types.GenerateContentResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error is not None:
            raise self.error
        return self.response


@dataclass
class FakeAsyncClient:
    models: FakeModelsClient
    closed: bool = False

    async def aclose(self) -> None:
        self.closed = True


@dataclass
class FakeGeminiClient:
    aio: FakeAsyncClient


def generated_response(text: str | None) -> types.GenerateContentResponse:
    if text is None:
        return types.GenerateContentResponse(candidates=[])
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part.from_text(text=text)])
            )
        ]
    )


def build_provider(
    response: types.GenerateContentResponse,
    *,
    model: str = "gemini-3.7-flash",
    error: errors.APIError | None = None,
) -> tuple[GeminiLLMProvider, FakeModelsClient, FakeAsyncClient]:
    models = FakeModelsClient(response=response, error=error)
    asynchronous_client = FakeAsyncClient(models=models)
    provider = GeminiLLMProvider(
        api_key="test-key",
        model=model,
        max_output_tokens=512,
        client=FakeGeminiClient(aio=asynchronous_client),
    )
    return provider, models, asynchronous_client


@pytest.mark.asyncio
async def test_gemini_three_uses_grounding_instruction_and_low_thinking() -> None:
    provider, models, _ = build_provider(generated_response("  Grounded answer.  "))

    answer = await provider.generate(
        system_instruction="Use context only.",
        content="Question and context",
    )

    assert answer == "Grounded answer."
    call = models.calls[0]
    assert call["model"] == "gemini-3.7-flash"
    assert call["contents"] == "Question and context"
    config = call["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.system_instruction == "Use context only."
    assert config.max_output_tokens == 512
    assert config.temperature is None
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW


@pytest.mark.asyncio
async def test_legacy_gemini_model_uses_low_temperature() -> None:
    provider, models, _ = build_provider(
        generated_response("Answer"),
        model="gemini-2.5-flash",
    )

    await provider.generate(system_instruction="Rules", content="Content")

    config = models.calls[0]["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.temperature == 0.1
    assert config.thinking_config is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("system_instruction", "content"),
    [("", "Content"), ("Rules", ""), ("  ", "Content"), ("Rules", "  ")],
)
async def test_blank_generation_input_is_rejected_before_api_call(
    system_instruction: str,
    content: str,
) -> None:
    provider, models, _ = build_provider(generated_response("Answer"))

    with pytest.raises(LLMGenerationError, match="must not be blank"):
        await provider.generate(
            system_instruction=system_instruction,
            content=content,
        )

    assert models.calls == []


@pytest.mark.asyncio
async def test_empty_model_response_is_rejected() -> None:
    provider, _, _ = build_provider(generated_response(None))

    with pytest.raises(LLMGenerationError, match="empty answer"):
        await provider.generate(system_instruction="Rules", content="Content")


@pytest.mark.asyncio
async def test_api_error_is_wrapped_without_provider_details() -> None:
    api_error = errors.APIError(500, {"error": {"message": "sensitive details"}})
    provider, _, _ = build_provider(generated_response(None), error=api_error)

    with pytest.raises(LLMGenerationError, match="generation failed") as caught:
        await provider.generate(system_instruction="Rules", content="Content")

    assert "sensitive details" not in str(caught.value)


@pytest.mark.asyncio
async def test_provider_closes_async_client() -> None:
    provider, _, asynchronous_client = build_provider(generated_response("Answer"))

    await provider.close()

    assert asynchronous_client.closed is True
