from dataclasses import dataclass, field
from math import isclose, nan

import pytest
from google.genai import errors, types

from app.core.constants import EMBEDDING_DIMENSION
from app.domain.errors import EmbeddingGenerationError
from app.infrastructure.ai.gemini_embeddings import GeminiEmbeddingProvider


@dataclass
class FakeModelsClient:
    response: types.EmbedContentResponse
    error: errors.APIError | None = None
    calls: list[dict[str, object]] = field(default_factory=list)

    async def embed_content(
        self,
        *,
        model: str,
        contents: object,
        config: types.EmbedContentConfig,
    ) -> types.EmbedContentResponse:
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


def embedding_response(*vectors: list[float]) -> types.EmbedContentResponse:
    return types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=vector) for vector in vectors]
    )


def build_provider(
    response: types.EmbedContentResponse,
    *,
    model: str = "gemini-embedding-2",
    dimension: int = 128,
    error: errors.APIError | None = None,
) -> tuple[GeminiEmbeddingProvider, FakeModelsClient, FakeAsyncClient]:
    models = FakeModelsClient(response=response, error=error)
    asynchronous_client = FakeAsyncClient(models=models)
    client = FakeGeminiClient(aio=asynchronous_client)
    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        model=model,
        dimension=dimension,
        client=client,
    )
    return provider, models, asynchronous_client


@pytest.mark.asyncio
async def test_embedding_two_formats_documents_as_separate_contents() -> None:
    vector = [0.1] * 128
    provider, models, _ = build_provider(embedding_response(vector, vector))

    embeddings = await provider.embed_documents(["First chunk", "Second chunk"])

    assert embeddings == (tuple(vector), tuple(vector))
    call = models.calls[0]
    contents = call["contents"]
    assert isinstance(contents, list)
    assert [content.parts[0].text for content in contents] == [
        "title: none | text: First chunk",
        "title: none | text: Second chunk",
    ]
    config = call["config"]
    assert isinstance(config, types.EmbedContentConfig)
    assert config.task_type is None
    assert config.output_dimensionality == 128


@pytest.mark.asyncio
async def test_embedding_two_formats_question_answering_query() -> None:
    vector = [0.2] * 128
    provider, models, _ = build_provider(embedding_response(vector))

    embedding = await provider.embed_query("How long is the warranty?")

    assert embedding == tuple(vector)
    contents = models.calls[0]["contents"]
    assert isinstance(contents, list)
    assert contents[0].parts[0].text == (
        "task: question answering | query: How long is the warranty?"
    )


@pytest.mark.asyncio
async def test_provider_requests_schema_embedding_dimension() -> None:
    vector = [0.2] * EMBEDDING_DIMENSION
    provider, models, _ = build_provider(
        embedding_response(vector),
        dimension=EMBEDDING_DIMENSION,
    )

    embedding = await provider.embed_query("How long is the warranty?")

    config = models.calls[0]["config"]
    assert isinstance(config, types.EmbedContentConfig)
    assert config.output_dimensionality == EMBEDDING_DIMENSION == 768
    assert len(embedding) == EMBEDDING_DIMENSION


@pytest.mark.asyncio
async def test_legacy_model_uses_task_types_and_normalizes_vectors() -> None:
    provider, models, _ = build_provider(
        embedding_response([3.0, 4.0] + [0.0] * 126),
        model="gemini-embedding-001",
    )

    embedding = await provider.embed_query("Warranty question")

    config = models.calls[0]["config"]
    assert isinstance(config, types.EmbedContentConfig)
    assert config.task_type == "QUESTION_ANSWERING"
    assert isclose(embedding[0], 0.6)
    assert isclose(embedding[1], 0.8)


@pytest.mark.asyncio
async def test_empty_document_batch_does_not_call_provider() -> None:
    provider, models, _ = build_provider(embedding_response())

    assert await provider.embed_documents([]) == ()
    assert models.calls == []


@pytest.mark.asyncio
async def test_blank_input_is_rejected_before_provider_call() -> None:
    provider, models, _ = build_provider(embedding_response())

    with pytest.raises(EmbeddingGenerationError, match="must not be blank"):
        await provider.embed_query("  ")

    assert models.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        types.EmbedContentResponse(embeddings=None),
        embedding_response(),
        embedding_response([0.1] * 127),
        embedding_response([nan] * 128),
    ],
)
async def test_invalid_provider_response_is_rejected(
    response: types.EmbedContentResponse,
) -> None:
    provider, _, _ = build_provider(response)

    with pytest.raises(EmbeddingGenerationError):
        await provider.embed_query("Question")


@pytest.mark.asyncio
async def test_provider_api_errors_are_wrapped() -> None:
    api_error = errors.APIError(500, {"error": {"message": "provider failure"}})
    provider, _, _ = build_provider(embedding_response(), error=api_error)

    with pytest.raises(EmbeddingGenerationError, match="request failed"):
        await provider.embed_query("Question")


@pytest.mark.asyncio
async def test_provider_closes_async_client() -> None:
    provider, _, asynchronous_client = build_provider(embedding_response())

    await provider.close()

    assert asynchronous_client.closed is True
