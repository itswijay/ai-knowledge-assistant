from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StringConstraints

from app.domain.entities.organization import MAX_ORGANIZATION_NAME_LENGTH

OrganizationName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_ORGANIZATION_NAME_LENGTH,
    ),
]


class CreateOrganizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: OrganizationName


class OrganizationResponse(BaseModel):
    id: UUID
    name: str
    created_at: datetime
    updated_at: datetime
