from collections.abc import Sequence
from math import isfinite

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.domain.entities import Document, DocumentChunk, RetrievedChunk
from app.domain.errors import VectorRepositoryError
from app.domain.types import EmbeddingVector
from app.infrastructure.database.models import DocumentChunkModel, DocumentModel
from app.infrastructure.database.session import AsyncSessionFactory


class PostgresVectorRepository:
    def __init__(
        self,
        session_factory: AsyncSessionFactory,
        *,
        embedding_dimension: int,
    ) -> None:
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be at least 1")
        self._session_factory = session_factory
        self._embedding_dimension = embedding_dimension

    async def save_document(
        self,
        document: Document,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        for chunk in chunks:
            if chunk.document_id != document.id:
                raise ValueError("Every chunk must belong to the supplied document")
            self._validate_embedding(chunk.embedding)

        document_model = DocumentModel(
            id=document.id,
            assistant_id=document.assistant_id,
            original_filename=document.original_filename,
            created_at=document.created_at,
            chunks=[self._to_chunk_model(chunk) for chunk in chunks],
        )

        try:
            async with self._session_factory() as session:
                async with session.begin():
                    session.add(document_model)
        except SQLAlchemyError as error:
            raise VectorRepositoryError("Unable to persist document vectors") from error

    async def search_similar(
        self,
        query_embedding: EmbeddingVector,
        *,
        limit: int,
        minimum_similarity: float,
    ) -> Sequence[RetrievedChunk]:
        self._validate_embedding(query_embedding)
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        if not isfinite(minimum_similarity) or not 0.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between 0 and 1")

        distance = DocumentChunkModel.embedding.cosine_distance(list(query_embedding))
        similarity = (1.0 - distance).label("similarity_score")
        statement = (
            select(DocumentChunkModel, DocumentModel.original_filename, similarity)
            .join(DocumentModel, DocumentChunkModel.document_id == DocumentModel.id)
            .where(similarity >= minimum_similarity)
            .order_by(distance.asc(), DocumentChunkModel.id.asc())
            .limit(limit)
        )

        try:
            async with self._session_factory() as session:
                result = await session.execute(statement)
        except SQLAlchemyError as error:
            raise VectorRepositoryError(
                "Unable to retrieve document vectors"
            ) from error

        return tuple(
            self._to_retrieved_chunk(
                chunk,
                original_filename,
                float(similarity_score),
            )
            for chunk, original_filename, similarity_score in result.all()
        )

    def _validate_embedding(self, embedding: EmbeddingVector) -> None:
        if len(embedding) != self._embedding_dimension:
            raise ValueError(
                f"Embedding must contain {self._embedding_dimension} values"
            )
        if not all(isfinite(value) for value in embedding):
            raise ValueError("Embedding values must be finite")
        if not any(value != 0.0 for value in embedding):
            raise ValueError("Embedding must not be a zero vector")

    @staticmethod
    def _to_chunk_model(chunk: DocumentChunk) -> DocumentChunkModel:
        return DocumentChunkModel(
            id=chunk.id,
            document_id=chunk.document_id,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            embedding=list(chunk.embedding),
            created_at=chunk.created_at,
        )

    @staticmethod
    def _to_retrieved_chunk(
        chunk: DocumentChunkModel,
        original_filename: str,
        similarity_score: float,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            original_filename=original_filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            similarity_score=similarity_score,
        )
