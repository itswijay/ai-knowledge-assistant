from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    assistant_id: UUID
    original_filename: str
    processed_page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)


class DocumentResponse(BaseModel):
    id: UUID
    assistant_id: UUID
    original_filename: str
    created_at: datetime
