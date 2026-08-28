from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest

from app.application.use_cases import CreateOrganizationCommand
from app.dependencies import (
    get_access_token_verifier,
    get_create_organization,
    get_get_organization,
    get_list_organizations,
)
from app.domain.entities import AuthenticatedUser, Organization
from app.domain.errors import (
    OrganizationNotFoundError,
    ResourceConflictError,
    TenantRepositoryError,
)
from app.main import create_app


@dataclass
class FakeAccessTokenVerifier:
    user: AuthenticatedUser
    tokens: list[str] = field(default_factory=list)

    async def verify(self, token: str) -> AuthenticatedUser:
        self.tokens.append(token)
        return self.user


@dataclass
class FakeCreateOrganization:
    organization: Organization | None = None
    error: Exception | None = None
    calls: list[CreateOrganizationCommand] = field(default_factory=list)

    async def execute(self, command: CreateOrganizationCommand) -> Organization:
        self.calls.append(command)
        if self.error is not None:
            raise self.error
        if self.organization is None:
            raise AssertionError("A fake organization is required")
        return self.organization


@dataclass
class FakeListOrganizations:
    organizations: Sequence[Organization] = ()
    error: Exception | None = None
    calls: list[UUID] = field(default_factory=list)

    async def execute(self, user_id: UUID) -> Sequence[Organization]:
        self.calls.append(user_id)
        if self.error is not None:
            raise self.error
        return self.organizations


@dataclass
class FakeGetOrganization:
    organization: Organization | None = None
    error: Exception | None = None
    calls: list[tuple[UUID, UUID]] = field(default_factory=list)

    async def execute(
        self,
        *,
        user_id: UUID,
        organization_id: UUID,
    ) -> Organization:
        self.calls.append((user_id, organization_id))
        if self.error is not None:
            raise self.error
        if self.organization is None:
            raise AssertionError("A fake organization is required")
        return self.organization


async def request_organization_endpoint(
    *,
    method: str,
    path: str,
    dependency: Callable[..., Any],
    use_case: object,
    user: AuthenticatedUser,
    json: object | None = None,
    authenticated: bool = True,
) -> tuple[httpx.Response, FakeAccessTokenVerifier]:
    application = create_app()
    verifier = FakeAccessTokenVerifier(user)

    async def override_verifier() -> FakeAccessTokenVerifier:
        return verifier

    async def override_use_case() -> object:
        return use_case

    application.dependency_overrides[get_access_token_verifier] = override_verifier
    application.dependency_overrides[dependency] = override_use_case
    headers = {"Authorization": "Bearer valid-token"} if authenticated else {}
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.request(
            method,
            path,
            headers=headers,
            json=json,
        )
    return response, verifier


@pytest.mark.asyncio
async def test_create_organization_uses_verified_identity() -> None:
    user = AuthenticatedUser(id=uuid4(), email="owner@example.com")
    organization = Organization(name="Acme")
    use_case = FakeCreateOrganization(organization=organization)

    response, verifier = await request_organization_endpoint(
        method="POST",
        path="/api/v1/organizations",
        dependency=get_create_organization,
        use_case=use_case,
        user=user,
        json={"name": "  Acme  "},
    )

    assert response.status_code == 201
    assert response.json()["id"] == str(organization.id)
    assert response.json()["name"] == "Acme"
    assert use_case.calls == [
        CreateOrganizationCommand(creator_user_id=user.id, name="Acme")
    ]
    assert verifier.tokens == ["valid-token"]


@pytest.mark.asyncio
async def test_list_organizations_uses_verified_identity() -> None:
    user = AuthenticatedUser(id=uuid4())
    organizations = (Organization(name="Acme"), Organization(name="Example"))
    use_case = FakeListOrganizations(organizations=organizations)

    response, _ = await request_organization_endpoint(
        method="GET",
        path="/api/v1/organizations",
        dependency=get_list_organizations,
        use_case=use_case,
        user=user,
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [
        str(organization.id) for organization in organizations
    ]
    assert [item["name"] for item in response.json()] == ["Acme", "Example"]
    assert use_case.calls == [user.id]


@pytest.mark.asyncio
async def test_get_organization_uses_verified_identity_and_path_id() -> None:
    user = AuthenticatedUser(id=uuid4())
    organization = Organization(name="Acme")
    use_case = FakeGetOrganization(organization=organization)

    response, _ = await request_organization_endpoint(
        method="GET",
        path=f"/api/v1/organizations/{organization.id}",
        dependency=get_get_organization,
        use_case=use_case,
        user=user,
    )

    assert response.status_code == 200
    assert response.json()["id"] == str(organization.id)
    assert use_case.calls == [(user.id, organization.id)]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["", "   ", "x" * 121, 42, None])
async def test_create_organization_rejects_invalid_name(name: object) -> None:
    use_case = FakeCreateOrganization()

    response, _ = await request_organization_endpoint(
        method="POST",
        path="/api/v1/organizations",
        dependency=get_create_organization,
        use_case=use_case,
        user=AuthenticatedUser(id=uuid4()),
        json={"name": name},
    )

    assert response.status_code == 422
    assert use_case.calls == []


@pytest.mark.asyncio
async def test_create_organization_rejects_client_supplied_owner_identity() -> None:
    use_case = FakeCreateOrganization()

    response, _ = await request_organization_endpoint(
        method="POST",
        path="/api/v1/organizations",
        dependency=get_create_organization,
        use_case=use_case,
        user=AuthenticatedUser(id=uuid4()),
        json={"name": "Acme", "creator_user_id": str(uuid4())},
    )

    assert response.status_code == 422
    assert use_case.calls == []


@pytest.mark.asyncio
async def test_organization_endpoints_require_authentication() -> None:
    use_case = FakeListOrganizations()

    response, verifier = await request_organization_endpoint(
        method="GET",
        path="/api/v1/organizations",
        dependency=get_list_organizations,
        use_case=use_case,
        user=AuthenticatedUser(id=uuid4()),
        authenticated=False,
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication credentials were not provided"}
    assert verifier.tokens == []
    assert use_case.calls == []


@pytest.mark.asyncio
async def test_cross_tenant_organization_is_concealed_as_not_found() -> None:
    use_case = FakeGetOrganization(
        error=OrganizationNotFoundError("Organization not found")
    )

    response, _ = await request_organization_endpoint(
        method="GET",
        path=f"/api/v1/organizations/{uuid4()}",
        dependency=get_get_organization,
        use_case=use_case,
        user=AuthenticatedUser(id=uuid4()),
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Organization not found"}


@pytest.mark.asyncio
async def test_organization_conflict_maps_to_safe_response() -> None:
    use_case = FakeCreateOrganization(
        error=ResourceConflictError("sensitive database constraint")
    )

    response, _ = await request_organization_endpoint(
        method="POST",
        path="/api/v1/organizations",
        dependency=get_create_organization,
        use_case=use_case,
        user=AuthenticatedUser(id=uuid4()),
        json={"name": "Acme"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Resource conflicts with existing data"}
    assert "sensitive" not in response.text


@pytest.mark.asyncio
async def test_tenant_repository_failure_maps_to_safe_response() -> None:
    use_case = FakeListOrganizations(
        error=TenantRepositoryError("sensitive connection details")
    )

    response, _ = await request_organization_endpoint(
        method="GET",
        path="/api/v1/organizations",
        dependency=get_list_organizations,
        use_case=use_case,
        user=AuthenticatedUser(id=uuid4()),
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Data service is temporarily unavailable"}
    assert "sensitive" not in response.text


@pytest.mark.asyncio
async def test_openapi_documents_organization_bearer_authentication() -> None:
    application = create_app()
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get("/openapi.json")

    paths = response.json()["paths"]
    assert paths["/api/v1/organizations"]["post"]["security"] == [{"HTTPBearer": []}]
    assert paths["/api/v1/organizations"]["get"]["security"] == [{"HTTPBearer": []}]
    assert paths["/api/v1/organizations/{organization_id}"]["get"]["security"] == [
        {"HTTPBearer": []}
    ]
