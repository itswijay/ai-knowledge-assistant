from dataclasses import dataclass, field
from uuid import uuid4

import httpx
import pytest

from app.application.use_cases import IngestDocumentCommand, IngestDocumentResult
from app.core.config import Settings, get_settings
from app.dependencies import get_ingest_document
from app.domain.errors import InvalidDocumentError
from app.main import create_app


@dataclass
class FakeIngestDocument:
    result: IngestDocumentResult | None = None
    error: Exception | None = None
    calls: list[IngestDocumentCommand] = field(default_factory=list)

    async def execute(self, command: IngestDocumentCommand) -> IngestDocumentResult:
        self.calls.append(command)
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise AssertionError("A fake ingestion result is required")
        return self.result


def build_test_settings(*, max_upload_size_mb: int = 1) -> Settings:
    return Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://postgres:password@localhost/app",
        gemini_api_key="test-api-key",
        max_upload_size_mb=max_upload_size_mb,
    )


async def post_document(
    fake_use_case: FakeIngestDocument,
    *,
    filename: str = "warranty.pdf",
    content: bytes = b"%PDF-test",
) -> httpx.Response:
    application = create_app()
    application.dependency_overrides[get_ingest_document] = lambda: fake_use_case
    application.dependency_overrides[get_settings] = lambda: build_test_settings()
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.post(
            "/api/v1/documents",
            files={"file": (filename, content, "application/pdf")},
        )


@pytest.mark.asyncio
async def test_document_upload_invokes_ingestion_and_serializes_result() -> None:
    document_id = uuid4()
    fake_use_case = FakeIngestDocument(
        result=IngestDocumentResult(
            document_id=document_id,
            original_filename="warranty.pdf",
            processed_page_count=2,
            chunk_count=4,
        )
    )

    response = await post_document(fake_use_case, filename="../warranty.pdf")

    assert response.status_code == 201
    assert response.json() == {
        "document_id": str(document_id),
        "original_filename": "warranty.pdf",
        "processed_page_count": 2,
        "chunk_count": 4,
    }
    assert fake_use_case.calls == [
        IngestDocumentCommand(filename="../warranty.pdf", content=b"%PDF-test")
    ]


@pytest.mark.asyncio
async def test_document_upload_reads_only_up_to_configured_limit() -> None:
    fake_use_case = FakeIngestDocument()
    oversized_content = b"%PDF-" + bytes(1024 * 1024)

    response = await post_document(fake_use_case, content=oversized_content)

    assert response.status_code == 413
    assert response.json() == {"detail": "PDF exceeds the 1 MB upload limit"}
    assert fake_use_case.calls == []


@pytest.mark.asyncio
async def test_document_processing_errors_return_unprocessable_content() -> None:
    fake_use_case = FakeIngestDocument(
        error=InvalidDocumentError("File content is not a valid PDF")
    )

    response = await post_document(fake_use_case, content=b"not-a-pdf")

    assert response.status_code == 422
    assert response.json() == {"detail": "File content is not a valid PDF"}


@pytest.mark.asyncio
async def test_document_upload_requires_multipart_file() -> None:
    application = create_app()
    application.dependency_overrides[get_ingest_document] = lambda: FakeIngestDocument()
    application.dependency_overrides[get_settings] = lambda: build_test_settings()
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.post("/api/v1/documents")

    assert response.status_code == 422
