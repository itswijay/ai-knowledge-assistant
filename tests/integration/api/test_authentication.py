from dataclasses import dataclass, field
from uuid import uuid4

import httpx
import pytest

from app.dependencies import get_access_token_verifier
from app.domain.entities import AuthenticatedUser
from app.domain.errors import (
    AccessTokenVerificationError,
    ExpiredAccessTokenError,
    InvalidAccessTokenError,
    MalformedAccessTokenError,
)
from app.main import create_app


@dataclass
class FakeAccessTokenVerifier:
    user: AuthenticatedUser | None = None
    error: Exception | None = None
    tokens: list[str] = field(default_factory=list)

    async def verify(self, token: str) -> AuthenticatedUser:
        self.tokens.append(token)
        if self.error is not None:
            raise self.error
        if self.user is None:
            raise AssertionError("A fake authenticated user is required")
        return self.user


async def get_me(
    verifier: FakeAccessTokenVerifier,
    *,
    authorization: str | None = "Bearer valid-access-token",
) -> httpx.Response:
    application = create_app()

    async def override_verifier() -> FakeAccessTokenVerifier:
        return verifier

    application.dependency_overrides[get_access_token_verifier] = override_verifier
    transport = httpx.ASGITransport(app=application)
    headers = {"Authorization": authorization} if authorization is not None else {}

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.get("/api/v1/me", headers=headers)


@pytest.mark.asyncio
async def test_me_returns_identity_from_verified_token() -> None:
    user_id = uuid4()
    verifier = FakeAccessTokenVerifier(
        user=AuthenticatedUser(id=user_id, email="user@example.com")
    )

    response = await get_me(verifier)

    assert response.status_code == 200
    assert response.json() == {"id": str(user_id), "email": "user@example.com"}
    assert verifier.tokens == ["valid-access-token"]


@pytest.mark.asyncio
async def test_me_returns_null_when_verified_token_has_no_email() -> None:
    user_id = uuid4()
    verifier = FakeAccessTokenVerifier(user=AuthenticatedUser(id=user_id))

    response = await get_me(verifier)

    assert response.status_code == 200
    assert response.json() == {"id": str(user_id), "email": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("authorization", [None, "Basic credentials", "Bearer"])
async def test_me_requires_bearer_credentials(authorization: str | None) -> None:
    verifier = FakeAccessTokenVerifier()

    response = await get_me(verifier, authorization=authorization)

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication credentials were not provided"}
    assert response.headers["www-authenticate"] == "Bearer"
    assert verifier.tokens == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verification_error", "expected_detail"),
    [
        (MalformedAccessTokenError("raw details"), "Access token is invalid"),
        (InvalidAccessTokenError("raw details"), "Access token is invalid"),
        (ExpiredAccessTokenError("raw details"), "Access token has expired"),
    ],
)
async def test_me_maps_invalid_credentials_to_unauthorized(
    verification_error: Exception,
    expected_detail: str,
) -> None:
    verifier = FakeAccessTokenVerifier(error=verification_error)

    response = await get_me(verifier)

    assert response.status_code == 401
    assert response.json() == {"detail": expected_detail}
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.asyncio
async def test_me_maps_jwks_failure_to_service_unavailable() -> None:
    verifier = FakeAccessTokenVerifier(
        error=AccessTokenVerificationError("sensitive provider details")
    )

    response = await get_me(verifier)

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Authentication service is temporarily unavailable"
    }
    assert "sensitive" not in response.text


@pytest.mark.asyncio
async def test_openapi_documents_bearer_authentication_for_me() -> None:
    application = create_app()
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get("/openapi.json")

    operation = response.json()["paths"]["/api/v1/me"]["get"]
    assert operation["security"] == [{"HTTPBearer": []}]
