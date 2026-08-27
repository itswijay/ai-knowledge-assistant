from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.application.use_cases import IngestDocument, IngestDocumentCommand
from app.dependencies import get_ingest_document, get_max_upload_size_bytes
from app.domain.errors import DocumentTooLargeError
from app.presentation.api.schemas.documents import DocumentUploadResponse

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF document to ingest")],
    use_case: Annotated[IngestDocument, Depends(get_ingest_document)],
    max_upload_size_bytes: Annotated[int, Depends(get_max_upload_size_bytes)],
) -> DocumentUploadResponse:
    content = await file.read(max_upload_size_bytes + 1)
    if len(content) > max_upload_size_bytes:
        size_mb = max_upload_size_bytes // (1024 * 1024)
        raise DocumentTooLargeError(f"PDF exceeds the {size_mb} MB upload limit")

    result = await use_case.execute(
        IngestDocumentCommand(
            filename=file.filename or "",
            content=content,
        )
    )
    return DocumentUploadResponse(
        document_id=result.document_id,
        original_filename=result.original_filename,
        processed_page_count=result.processed_page_count,
        chunk_count=result.chunk_count,
    )
