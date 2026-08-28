from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Response, UploadFile, status

from app.application.use_cases import (
    DeleteDocument,
    IngestDocument,
    IngestDocumentCommand,
    ListDocuments,
)
from app.dependencies import (
    get_authenticated_user,
    get_delete_document,
    get_ingest_document,
    get_list_documents,
    get_max_upload_size_bytes,
)
from app.domain.entities import AuthenticatedUser, Document
from app.domain.errors import DocumentTooLargeError
from app.presentation.api.schemas.documents import (
    DocumentResponse,
    DocumentUploadResponse,
)

router = APIRouter(prefix="/api/v1", tags=["documents"])


@router.post(
    "/assistants/{assistant_id}/documents",
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


@router.get(
    "/assistants/{assistant_id}/documents",
    response_model=list[DocumentResponse],
)
async def list_documents(
    assistant_id: UUID,
    user: Annotated[AuthenticatedUser, Depends(get_authenticated_user)],
    use_case: Annotated[ListDocuments, Depends(get_list_documents)],
) -> list[DocumentResponse]:
    documents = await use_case.execute(
        user_id=user.id,
        assistant_id=assistant_id,
    )
    return [_to_response(document) for document in documents]


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(
    document_id: UUID,
    user: Annotated[AuthenticatedUser, Depends(get_authenticated_user)],
    use_case: Annotated[DeleteDocument, Depends(get_delete_document)],
) -> Response:
    await use_case.execute(user_id=user.id, document_id=document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _to_response(document: Document) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        assistant_id=document.assistant_id,
        original_filename=document.original_filename,
        created_at=document.created_at,
    )
