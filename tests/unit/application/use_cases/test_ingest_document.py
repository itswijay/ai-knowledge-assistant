from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from app.application.errors import NoExtractableTextError
from app.application.services import WordChunker
from app.application.use_cases import IngestDocument, IngestDocumentCommand
from app.domain.entities import (
    Assistant,
    Document,
    DocumentChunk,
    DocumentPage,
    RetrievedChunk,
)
from app.domain.errors import (
    AssistantNotFoundError,
    EmbeddingGenerationError,
    InsufficientPermissionError,
)
from app.domain.types import EmbeddingVector


@dataclass
class FakeDocumentValidator:
    safe_filename: str = "warranty.pdf"
    calls: list[tuple[str, bytes]] = field(default_factory=list)

    def validate(self, filename: str, content: bytes) -> str:
        self.calls.append((filename, content))
        return self.safe_filename


@dataclass
class FakeDocumentParser:
    pages: Sequence[DocumentPage]
    calls: list[bytes] = field(default_factory=list)

    def parse(self, content: bytes) -> Sequence[DocumentPage]:
        self.calls.append(content)
        return self.pages


@dataclass
class FakeEmbeddingProvider:
    embeddings: Sequence[EmbeddingVector] | None = None
    document_calls: list[Sequence[str]] = field(default_factory=list)

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[EmbeddingVector]:
        self.document_calls.append(texts)
        if self.embeddings is not None:
            return self.embeddings
        return tuple((float(index + 1),) for index, _ in enumerate(texts))

    async def embed_query(self, text: str) -> EmbeddingVector:
        raise AssertionError("Query embedding is not part of document ingestion")


@dataclass
class FakeVectorRepository:
    saved: list[tuple[Document, Sequence[DocumentChunk]]] = field(default_factory=list)

    async def save_document(
        self,
        document: Document,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        self.saved.append((document, chunks))

    async def search_similar(
        self,
        query_embedding: EmbeddingVector,
        *,
        limit: int,
        minimum_similarity: float,
    ) -> Sequence[RetrievedChunk]:
        raise AssertionError("Vector search is not part of document ingestion")


@dataclass
class FakeAssistantAccessChecker:
    assistant: Assistant
    error: Exception | None = None
    calls: list[tuple[UUID, UUID]] = field(default_factory=list)

    async def require_manager(
        self,
        *,
        user_id: UUID,
        assistant_id: UUID,
    ) -> Assistant:
        self.calls.append((user_id, assistant_id))
        if self.error is not None:
            raise self.error
        return self.assistant


def build_use_case(
    *,
    pages: Sequence[DocumentPage],
    embeddings: Sequence[EmbeddingVector] | None = None,
) -> tuple[
    IngestDocument,
    FakeDocumentValidator,
    FakeDocumentParser,
    FakeEmbeddingProvider,
    FakeVectorRepository,
    FakeAssistantAccessChecker,
]:
    validator = FakeDocumentValidator()
    parser = FakeDocumentParser(pages=pages)
    embedding_provider = FakeEmbeddingProvider(embeddings=embeddings)
    repository = FakeVectorRepository()
    access_checker = FakeAssistantAccessChecker(
        Assistant(organization_id=uuid4(), name="Support")
    )
    use_case = IngestDocument(
        validator=validator,
        parser=parser,
        chunker=WordChunker(chunk_size_words=3, overlap_words=1),
        embedding_provider=embedding_provider,
        vector_repository=repository,
        assistant_access_checker=access_checker,
    )
    return use_case, validator, parser, embedding_provider, repository, access_checker


@pytest.mark.asyncio
async def test_ingestion_orchestrates_processing_and_persistence() -> None:
    pages = (
        DocumentPage(page_number=1, content="one two three four"),
        DocumentPage(page_number=3, content="five six seven"),
    )
    use_case, validator, parser, embeddings, repository, access_checker = (
        build_use_case(pages=pages)
    )
    content = b"%PDF-test"
    user_id = uuid4()
    assistant_id = access_checker.assistant.id

    result = await use_case.execute(
        IngestDocumentCommand(
            user_id=user_id,
            assistant_id=assistant_id,
            filename="../unsafe.pdf",
            content=content,
        )
    )

    assert access_checker.calls == [(user_id, assistant_id)]
    assert validator.calls == [("../unsafe.pdf", content)]
    assert parser.calls == [content]
    assert embeddings.document_calls == [
        ["one two three", "three four", "five six seven"]
    ]
    assert result.original_filename == "warranty.pdf"
    assert result.assistant_id == assistant_id
    assert result.processed_page_count == 2
    assert result.chunk_count == 3
    assert len(repository.saved) == 1

    stored_document, stored_chunks = repository.saved[0]
    assert stored_document.id == result.document_id
    assert stored_document.assistant_id == assistant_id
    assert stored_document.original_filename == "warranty.pdf"
    assert [chunk.document_id for chunk in stored_chunks] == [
        stored_document.id,
        stored_document.id,
        stored_document.id,
    ]
    assert all(not hasattr(chunk, "original_filename") for chunk in stored_chunks)
    assert [chunk.page_number for chunk in stored_chunks] == [1, 1, 3]
    assert [chunk.chunk_index for chunk in stored_chunks] == [0, 1, 2]
    assert [chunk.embedding for chunk in stored_chunks] == [(1.0,), (2.0,), (3.0,)]


@pytest.mark.asyncio
async def test_no_extractable_pages_stops_before_embedding_and_persistence() -> None:
    use_case, _, _, embeddings, repository, access_checker = build_use_case(pages=[])

    with pytest.raises(NoExtractableTextError, match="no extractable text"):
        await use_case.execute(
            IngestDocumentCommand(
                user_id=uuid4(),
                assistant_id=access_checker.assistant.id,
                filename="empty.pdf",
                content=b"%PDF-empty",
            )
        )

    assert embeddings.document_calls == []
    assert repository.saved == []


@pytest.mark.asyncio
async def test_embedding_count_mismatch_stops_before_persistence() -> None:
    pages = [DocumentPage(page_number=1, content="one two three four")]
    use_case, _, _, _, repository, access_checker = build_use_case(
        pages=pages,
        embeddings=[(0.1,)],
    )

    with pytest.raises(EmbeddingGenerationError, match="unexpected number"):
        await use_case.execute(
            IngestDocumentCommand(
                user_id=uuid4(),
                assistant_id=access_checker.assistant.id,
                filename="document.pdf",
                content=b"%PDF-test",
            )
        )

    assert repository.saved == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        InsufficientPermissionError("Owner or admin role required"),
        AssistantNotFoundError("Assistant not found"),
    ],
)
async def test_unauthorized_ingestion_stops_before_file_processing(
    error: Exception,
) -> None:
    use_case, validator, parser, embeddings, repository, access_checker = (
        build_use_case(
            pages=[DocumentPage(page_number=1, content="content")],
        )
    )
    access_checker.error = error

    with pytest.raises(type(error), match=str(error)):
        await use_case.execute(
            IngestDocumentCommand(
                user_id=uuid4(),
                assistant_id=access_checker.assistant.id,
                filename="private.pdf",
                content=b"%PDF-private",
            )
        )

    assert validator.calls == []
    assert parser.calls == []
    assert embeddings.document_calls == []
    assert repository.saved == []
