from dataclasses import dataclass
from uuid import UUID

from app.application.errors import NoExtractableTextError
from app.application.services import AssistantAccessChecker, TextChunker
from app.domain.entities import Document, DocumentChunk
from app.domain.errors import EmbeddingGenerationError
from app.domain.ports import (
    DocumentParser,
    DocumentValidator,
    EmbeddingProvider,
    VectorRepository,
)


@dataclass(frozen=True, slots=True)
class IngestDocumentCommand:
    user_id: UUID
    assistant_id: UUID
    filename: str
    content: bytes


@dataclass(frozen=True, slots=True)
class IngestDocumentResult:
    document_id: UUID
    assistant_id: UUID
    original_filename: str
    processed_page_count: int
    chunk_count: int


class IngestDocument:
    def __init__(
        self,
        *,
        validator: DocumentValidator,
        parser: DocumentParser,
        chunker: TextChunker,
        embedding_provider: EmbeddingProvider,
        vector_repository: VectorRepository,
        assistant_access_checker: AssistantAccessChecker,
    ) -> None:
        self._validator = validator
        self._parser = parser
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vector_repository = vector_repository
        self._assistant_access_checker = assistant_access_checker

    async def execute(self, command: IngestDocumentCommand) -> IngestDocumentResult:
        await self._assistant_access_checker.require_manager(
            user_id=command.user_id,
            assistant_id=command.assistant_id,
        )
        safe_filename = self._validator.validate(command.filename, command.content)
        pages = tuple(self._parser.parse(command.content))
        if not pages:
            raise NoExtractableTextError(
                "PDF contains no extractable text; OCR is not supported."
            )

        chunk_drafts = tuple(self._chunker.chunk_pages(pages))
        if not chunk_drafts:
            raise NoExtractableTextError("PDF produced no ingestible text chunks")

        embeddings = tuple(
            await self._embedding_provider.embed_documents(
                [chunk.content for chunk in chunk_drafts]
            )
        )
        if len(embeddings) != len(chunk_drafts):
            raise EmbeddingGenerationError(
                "Embedding provider returned an unexpected number of vectors"
            )

        document = Document(
            assistant_id=command.assistant_id,
            original_filename=safe_filename,
        )
        chunks = tuple(
            DocumentChunk(
                document_id=document.id,
                page_number=draft.page_number,
                chunk_index=draft.chunk_index,
                content=draft.content,
                embedding=embedding,
            )
            for draft, embedding in zip(chunk_drafts, embeddings, strict=True)
        )
        await self._vector_repository.save_document(document, chunks)

        return IngestDocumentResult(
            document_id=document.id,
            assistant_id=document.assistant_id,
            original_filename=document.original_filename,
            processed_page_count=len(pages),
            chunk_count=len(chunks),
        )
