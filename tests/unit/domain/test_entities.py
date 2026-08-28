from collections.abc import Callable
from dataclasses import FrozenInstanceError
from datetime import UTC
from math import inf, nan
from uuid import UUID, uuid4

import pytest

from app.domain.entities import (
    Answer,
    Document,
    DocumentChunk,
    DocumentPage,
    RetrievedChunk,
    SourceReference,
)


def test_document_has_generated_identity_and_utc_timestamp() -> None:
    assistant_id = uuid4()
    document = Document(
        assistant_id=assistant_id,
        original_filename="handbook.pdf",
    )

    assert document.id
    assert document.assistant_id == assistant_id
    assert document.created_at.tzinfo is UTC


def test_document_is_immutable() -> None:
    document = Document(assistant_id=uuid4(), original_filename="handbook.pdf")

    with pytest.raises(FrozenInstanceError):
        document.original_filename = "changed.pdf"  # type: ignore[misc]


@pytest.mark.parametrize("filename", ["", "   "])
def test_document_rejects_blank_filename(filename: str) -> None:
    with pytest.raises(ValueError, match="original_filename"):
        Document(assistant_id=uuid4(), original_filename=filename)


def test_document_rejects_nil_assistant_id() -> None:
    with pytest.raises(ValueError, match="assistant_id"):
        Document(assistant_id=UUID(int=0), original_filename="handbook.pdf")


@pytest.mark.parametrize(
    ("page_number", "content"),
    [(0, "content"), (1, ""), (1, "  ")],
)
def test_document_page_enforces_valid_page_content(
    page_number: int,
    content: str,
) -> None:
    with pytest.raises(ValueError):
        DocumentPage(page_number=page_number, content=content)


def test_document_chunk_preserves_ownership_and_page_metadata() -> None:
    document_id = uuid4()
    chunk = DocumentChunk(
        document_id=document_id,
        page_number=3,
        chunk_index=2,
        content="Warranty coverage lasts two years.",
        embedding=(0.1, 0.2, 0.3),
    )

    assert chunk.document_id == document_id
    assert chunk.page_number == 3
    assert chunk.chunk_index == 2


@pytest.mark.parametrize("embedding", [(), (nan,), (inf,), (-inf,)])
def test_document_chunk_rejects_invalid_embedding(
    embedding: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="embedding"):
        DocumentChunk(
            document_id=uuid4(),
            page_number=1,
            chunk_index=0,
            content="Content",
            embedding=embedding,
        )


@pytest.mark.parametrize("similarity_score", [-1.01, 1.01, nan, inf])
def test_retrieved_chunk_rejects_invalid_similarity(
    similarity_score: float,
) -> None:
    with pytest.raises(ValueError, match="similarity_score"):
        RetrievedChunk(
            chunk_id=uuid4(),
            document_id=uuid4(),
            original_filename="handbook.pdf",
            page_number=1,
            chunk_index=0,
            content="Content",
            similarity_score=similarity_score,
        )


def test_answer_contains_typed_source_references() -> None:
    source = SourceReference(document="handbook.pdf", page=3)
    answer = Answer(text="Coverage lasts two years.", sources=(source,))

    assert answer.sources == (source,)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SourceReference(document="", page=1),
        lambda: SourceReference(document="handbook.pdf", page=0),
        lambda: Answer(text=""),
    ],
)
def test_answer_models_reject_invalid_values(factory: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        factory()
