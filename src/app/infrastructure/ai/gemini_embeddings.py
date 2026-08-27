from collections.abc import Sequence
from math import isfinite, sqrt
from typing import Protocol

from google import genai
from google.genai import errors, types

from app.domain.errors import EmbeddingGenerationError
from app.domain.types import EmbeddingVector

GEMINI_EMBEDDING_2_PREFIX = "gemini-embedding-2"
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "QUESTION_ANSWERING"


class AsyncModelsClient(Protocol):
    async def embed_content(
        self,
        *,
        model: str,
        contents: object,
        config: types.EmbedContentConfig,
    ) -> types.EmbedContentResponse: ...


class AsyncGeminiClient(Protocol):
    models: AsyncModelsClient

    async def aclose(self) -> None: ...


class GeminiClient(Protocol):
    aio: AsyncGeminiClient


class GeminiEmbeddingProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimension: int,
        client: GeminiClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        if not model.strip():
            raise ValueError("model must not be blank")
        if not 128 <= dimension <= 3072:
            raise ValueError("dimension must be between 128 and 3072")

        self._model = model
        self._dimension = dimension
        self._client = client or genai.Client(api_key=api_key)

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[EmbeddingVector]:
        if not texts:
            return ()
        return await self._embed(texts, task_type=DOCUMENT_TASK_TYPE, query=False)

    async def embed_query(self, text: str) -> EmbeddingVector:
        embeddings = await self._embed((text,), task_type=QUERY_TASK_TYPE, query=True)
        return embeddings[0]

    async def close(self) -> None:
        await self._client.aio.aclose()

    async def _embed(
        self,
        texts: Sequence[str],
        *,
        task_type: str,
        query: bool,
    ) -> tuple[EmbeddingVector, ...]:
        if any(not text.strip() for text in texts):
            raise EmbeddingGenerationError("Embedding input must not be blank")

        embedding_2 = self._model.startswith(GEMINI_EMBEDDING_2_PREFIX)
        contents = (
            self._prepare_contents(texts, query=query) if embedding_2 else list(texts)
        )
        config = types.EmbedContentConfig(
            output_dimensionality=self._dimension,
            task_type=None if embedding_2 else task_type,
        )

        try:
            response = await self._client.aio.models.embed_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except errors.APIError as error:
            raise EmbeddingGenerationError("Gemini embedding request failed") from error

        embeddings = self._extract_embeddings(response, expected_count=len(texts))
        if not embedding_2 and self._dimension != 3072:
            return tuple(self._normalize(embedding) for embedding in embeddings)
        return embeddings

    def _prepare_contents(
        self,
        texts: Sequence[str],
        *,
        query: bool,
    ) -> list[types.Content]:
        if query:
            prepared = [f"task: question answering | query: {text}" for text in texts]
        else:
            prepared = [f"title: none | text: {text}" for text in texts]
        return [
            types.Content(parts=[types.Part.from_text(text=text)]) for text in prepared
        ]

    def _extract_embeddings(
        self,
        response: types.EmbedContentResponse,
        *,
        expected_count: int,
    ) -> tuple[EmbeddingVector, ...]:
        if response.embeddings is None or len(response.embeddings) != expected_count:
            raise EmbeddingGenerationError(
                "Gemini returned an unexpected number of embeddings"
            )

        vectors: list[EmbeddingVector] = []
        for embedding in response.embeddings:
            values = embedding.values
            if values is None or len(values) != self._dimension:
                raise EmbeddingGenerationError(
                    "Gemini returned an embedding with an unexpected dimension"
                )
            vector = tuple(values)
            if not all(isfinite(value) for value in vector):
                raise EmbeddingGenerationError(
                    "Gemini returned non-finite embedding values"
                )
            vectors.append(vector)
        return tuple(vectors)

    @staticmethod
    def _normalize(embedding: EmbeddingVector) -> EmbeddingVector:
        magnitude = sqrt(sum(value * value for value in embedding))
        if magnitude == 0:
            raise EmbeddingGenerationError("Gemini returned a zero-length embedding")
        return tuple(value / magnitude for value in embedding)
