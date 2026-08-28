from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.application.use_cases import IngestDocument, IngestDocumentCommand
from app.dependencies import (
    get_authenticated_user,
    get_ingest_document,
    get_max_upload_size_bytes,
)
from app.domain.entities import AuthenticatedUser
from app.domain.errors import DocumentTooLargeError
from app.presentation.api.schemas.documents import DocumentUploadResponse

router = APIRouter(prefix="/api/v1/assistants", tags=["documents"])


@router.post(
    "/{assistant_id}/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    assistant_id: UUID,
    file: Annotated[UploadFile, File(description="PDF document to ingest")],
    user: Annotated[AuthenticatedUser, Depends(get_authenticated_user)],
    use_case: Annotated[IngestDocument, Depends(get_ingest_document)],
    max_upload_size_bytes: Annotated[int, Depends(get_max_upload_size_bytes)],
) -> DocumentUploadResponse:
    content = await file.read(max_upload_size_bytes + 1)
    if len(content) > max_upload_size_bytes:
        size_mb = max_upload_size_bytes // (1024 * 1024)
        raise DocumentTooLargeError(f"PDF exceeds the {size_mb} MB upload limit")

    result = await use_case.execute(
        IngestDocumentCommand(
            user_id=user.id,
            assistant_id=assistant_id,
            filename=file.filename or "",
            content=content,
        )
    )
    return DocumentUploadResponse(
        document_id=result.document_id,
        assistant_id=result.assistant_id,
        original_filename=result.original_filename,
        processed_page_count=result.processed_page_count,
        chunk_count=result.chunk_count,
    )
