from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.database.base import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


class DocumentModel(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "length(btrim(original_filename)) > 0",
            name="original_filename_not_blank",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.now(),
    )
    chunks: Mapped[list[DocumentChunkModel]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class DocumentChunkModel(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        CheckConstraint(
            "length(btrim(original_filename)) > 0",
            name="original_filename_not_blank",
        ),
        CheckConstraint("page_number >= 1", name="page_number_positive"),
        CheckConstraint("chunk_index >= 0", name="chunk_index_not_negative"),
        CheckConstraint("length(btrim(content)) > 0", name="content_not_blank"),
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="document_chunk_index_unique",
        ),
        Index("ix_document_chunks_document_id", "document_id"),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    document_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    page_number: Mapped[int] = mapped_column(nullable=False)
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.now(),
    )
    document: Mapped[DocumentModel] = relationship(back_populates="chunks")
