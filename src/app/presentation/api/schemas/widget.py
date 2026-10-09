from uuid import UUID

from pydantic import BaseModel, Field


class WidgetConfigResponse(BaseModel):
    id: UUID
    name: str = Field(min_length=1)
    welcome_message: str = Field(min_length=1)
    logo_url: str | None = None
    primary_color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
