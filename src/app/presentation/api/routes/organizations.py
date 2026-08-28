from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from app.application.use_cases import (
    CreateOrganization,
    CreateOrganizationCommand,
    GetOrganization,
    ListOrganizations,
)
from app.dependencies import (
    get_authenticated_user,
    get_create_organization,
    get_get_organization,
    get_list_organizations,
)
from app.domain.entities import AuthenticatedUser, Organization
from app.presentation.api.schemas.organizations import (
    CreateOrganizationRequest,
    OrganizationResponse,
)

router = APIRouter(prefix="/api/v1/organizations", tags=["organizations"])


@router.post(
    "",
    response_model=OrganizationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_organization(
    request: CreateOrganizationRequest,
    user: Annotated[AuthenticatedUser, Depends(get_authenticated_user)],
    use_case: Annotated[CreateOrganization, Depends(get_create_organization)],
) -> OrganizationResponse:
    organization = await use_case.execute(
        CreateOrganizationCommand(
            creator_user_id=user.id,
            name=request.name,
        )
    )
    return _to_response(organization)


@router.get("", response_model=list[OrganizationResponse])
async def list_organizations(
    user: Annotated[AuthenticatedUser, Depends(get_authenticated_user)],
    use_case: Annotated[ListOrganizations, Depends(get_list_organizations)],
) -> list[OrganizationResponse]:
    organizations = await use_case.execute(user.id)
    return [_to_response(organization) for organization in organizations]


@router.get("/{organization_id}", response_model=OrganizationResponse)
async def get_organization(
    organization_id: UUID,
    user: Annotated[AuthenticatedUser, Depends(get_authenticated_user)],
    use_case: Annotated[GetOrganization, Depends(get_get_organization)],
) -> OrganizationResponse:
    organization = await use_case.execute(
        user_id=user.id,
        organization_id=organization_id,
    )
    return _to_response(organization)


def _to_response(organization: Organization) -> OrganizationResponse:
    return OrganizationResponse(
        id=organization.id,
        name=organization.name,
        created_at=organization.created_at,
        updated_at=organization.updated_at,
    )
