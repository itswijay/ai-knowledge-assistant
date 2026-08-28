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

from app.core.constants import EMBEDDING_DIMENSION
from app.domain.entities.assistant import (
    DEFAULT_ASSISTANT_INSTRUCTIONS,
    DEFAULT_PRIMARY_COLOR,
    DEFAULT_WELCOME_MESSAGE,
)
from app.infrastructure.database.base import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


class OrganizationModel(Base):
    __tablename__ = "organizations"
    __table_args__ = (
        CheckConstraint("length(btrim(name)) > 0", name="name_not_blank"),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
        server_default=func.now(),
    )
    members: Mapped[list[OrganizationMemberModel]] = relationship(
        back_populates="organization",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    assistants: Mapped[list[AssistantModel]] = relationship(
        back_populates="organization",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class OrganizationMemberModel(Base):
    __tablename__ = "organization_members"
    __table_args__ = (
        CheckConstraint(
            "role IN ('owner', 'admin', 'member')",
            name="role_valid",
        ),
        Index("ix_organization_members_organization_id", "organization_id"),
        Index("ix_organization_members_user_id", "user_id"),
    )

    organization_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    # The migration enforces the auth.users foreign key. It is intentionally
    # omitted from ORM metadata so SQLAlchemy does not try to manage auth.users.
    user_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.now(),
    )
    organization: Mapped[OrganizationModel] = relationship(back_populates="members")


class AssistantModel(Base):
    __tablename__ = "assistants"
    __table_args__ = (
        CheckConstraint("length(btrim(name)) > 0", name="name_not_blank"),
        CheckConstraint(
            "description IS NULL OR length(btrim(description)) > 0",
            name="description_not_blank",
        ),
        CheckConstraint(
            "length(btrim(welcome_message)) > 0",
            name="welcome_message_not_blank",
        ),
        CheckConstraint(
            "length(btrim(assistant_instructions)) > 0",
            name="assistant_instructions_not_blank",
        ),
        CheckConstraint(
            "logo_url IS NULL OR length(btrim(logo_url)) > 0",
            name="logo_url_not_blank",
        ),
        CheckConstraint(
            "primary_color ~ '^#[0-9A-Fa-f]{6}$'",
            name="primary_color_valid",
        ),
        Index("ix_assistants_organization_id", "organization_id"),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    organization_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    welcome_message: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        default=DEFAULT_WELCOME_MESSAGE,
        server_default=DEFAULT_WELCOME_MESSAGE,
    )
    assistant_instructions: Mapped[str] = mapped_column(
        String(4000),
        nullable=False,
        default=DEFAULT_ASSISTANT_INSTRUCTIONS,
        server_default=DEFAULT_ASSISTANT_INSTRUCTIONS,
    )
    logo_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    primary_color: Mapped[str] = mapped_column(
        String(7),
        nullable=False,
        default=DEFAULT_PRIMARY_COLOR,
        server_default=DEFAULT_PRIMARY_COLOR,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
        server_default=func.now(),
    )
    organization: Mapped[OrganizationModel] = relationship(back_populates="assistants")
    documents: Mapped[list[DocumentModel]] = relationship(
        back_populates="assistant",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class DocumentModel(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "length(btrim(original_filename)) > 0",
            name="original_filename_not_blank",
        ),
        Index("ix_documents_assistant_id", "assistant_id"),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    assistant_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("assistants.id", ondelete="CASCADE"),
        nullable=False,
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
    assistant: Mapped[AssistantModel] = relationship(back_populates="documents")


class DocumentChunkModel(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
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
    page_number: Mapped[int] = mapped_column(nullable=False)
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSION),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.now(),
    )
    document: Mapped[DocumentModel] = relationship(back_populates="chunks")
