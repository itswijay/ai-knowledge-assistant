from uuid import UUID

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    assistant_id: UUID
    original_filename: str
    processed_page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
