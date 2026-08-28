from dataclasses import dataclass, field
from uuid import UUID, uuid4

import httpx
import pytest

from app.application.use_cases import IngestDocumentCommand, IngestDocumentResult
from app.dependencies import (
    get_access_token_verifier,
    get_ingest_document,
    get_max_upload_size_bytes,
)
from app.domain.entities import AuthenticatedUser
from app.domain.errors import InsufficientPermissionError, InvalidDocumentError
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


@dataclass
class FakeAccessTokenVerifier:
    user: AuthenticatedUser
    tokens: list[str] = field(default_factory=list)

    async def verify(self, token: str) -> AuthenticatedUser:
        self.tokens.append(token)
        return self.user


async def post_document(
    fake_use_case: FakeIngestDocument,
    *,
    user: AuthenticatedUser | None = None,
    assistant_id: UUID | None = None,
    filename: str = "warranty.pdf",
    content: bytes = b"%PDF-test",
    authenticated: bool = True,
) -> httpx.Response:
    application = create_app()
    authenticated_user = user or AuthenticatedUser(id=uuid4())
    target_assistant_id = assistant_id or uuid4()
    verifier = FakeAccessTokenVerifier(authenticated_user)

    async def override_verifier() -> FakeAccessTokenVerifier:
        return verifier

    async def override_use_case() -> FakeIngestDocument:
        return fake_use_case

    async def override_max_upload_size_bytes() -> int:
        return 1024 * 1024

    application.dependency_overrides[get_ingest_document] = override_use_case
    application.dependency_overrides[get_access_token_verifier] = override_verifier
    application.dependency_overrides[get_max_upload_size_bytes] = (
        override_max_upload_size_bytes
    )
    transport = httpx.ASGITransport(app=application)
    headers = {"Authorization": "Bearer valid-token"} if authenticated else {}

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.post(
            f"/api/v1/assistants/{target_assistant_id}/documents",
            headers=headers,
            files={"file": (filename, content, "application/pdf")},
        )


@pytest.mark.asyncio
async def test_document_upload_invokes_ingestion_and_serializes_result() -> None:
    document_id = uuid4()
    assistant_id = uuid4()
    user = AuthenticatedUser(id=uuid4())
    fake_use_case = FakeIngestDocument(
        result=IngestDocumentResult(
            document_id=document_id,
            assistant_id=assistant_id,
            original_filename="warranty.pdf",
            processed_page_count=2,
            chunk_count=4,
        )
    )

    response = await post_document(
        fake_use_case,
        user=user,
        assistant_id=assistant_id,
        filename="../warranty.pdf",
    )

    assert response.status_code == 201
    assert response.json() == {
        "document_id": str(document_id),
        "assistant_id": str(assistant_id),
        "original_filename": "warranty.pdf",
        "processed_page_count": 2,
        "chunk_count": 4,
    }
    assert fake_use_case.calls == [
        IngestDocumentCommand(
            user_id=user.id,
            assistant_id=assistant_id,
            filename="../warranty.pdf",
            content=b"%PDF-test",
        )
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
async def test_document_upload_requires_authentication() -> None:
    fake_use_case = FakeIngestDocument()

    response = await post_document(fake_use_case, authenticated=False)

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication credentials were not provided"}
    assert fake_use_case.calls == []


@pytest.mark.asyncio
async def test_document_upload_maps_member_role_to_forbidden() -> None:
    fake_use_case = FakeIngestDocument(
        error=InsufficientPermissionError("Owner or admin role required")
    )

    response = await post_document(fake_use_case)

    assert response.status_code == 403
    assert response.json() == {"detail": "Owner or admin role required"}


@pytest.mark.asyncio
async def test_document_upload_requires_multipart_file() -> None:
    application = create_app()
    verifier = FakeAccessTokenVerifier(AuthenticatedUser(id=uuid4()))

    async def override_verifier() -> FakeAccessTokenVerifier:
        return verifier

    async def override_use_case() -> FakeIngestDocument:
        return FakeIngestDocument()

    async def override_max_upload_size_bytes() -> int:
        return 1024 * 1024

    application.dependency_overrides[get_ingest_document] = override_use_case
    application.dependency_overrides[get_access_token_verifier] = override_verifier
    application.dependency_overrides[get_max_upload_size_bytes] = (
        override_max_upload_size_bytes
    )
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            f"/api/v1/assistants/{uuid4()}/documents",
            headers={"Authorization": "Bearer valid-token"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_openapi_documents_upload_bearer_authentication() -> None:
    application = create_app()
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get("/openapi.json")

    operation = response.json()["paths"]["/api/v1/assistants/{assistant_id}/documents"][
        "post"
    ]
    assert operation["security"] == [{"HTTPBearer": []}]
    assert "/api/v1/documents" not in response.json()["paths"]
