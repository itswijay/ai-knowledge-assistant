from dataclasses import FrozenInstanceError
from uuid import UUID, uuid4

import pytest

from app.domain.entities import AuthenticatedUser
from app.domain.errors import (
    AuthenticationError,
    ExpiredAccessTokenError,
    InvalidAccessTokenError,
    MalformedAccessTokenError,
    MissingAccessTokenError,
)
from app.domain.ports import AccessTokenVerifier


class FakeAccessTokenVerifier:
    def __init__(self, user: AuthenticatedUser) -> None:
        self.user = user
        self.tokens: list[str] = []

    async def verify(self, token: str) -> AuthenticatedUser:
        self.tokens.append(token)
        return self.user


def accepts_access_token_verifier(verifier: AccessTokenVerifier) -> None:
    assert verifier is not None


def test_authenticated_user_preserves_verified_identity() -> None:
    user_id = uuid4()

    user = AuthenticatedUser(id=user_id, email="user@example.com")

    assert user.id == user_id
    assert user.email == "user@example.com"


def test_authenticated_user_is_immutable() -> None:
    user = AuthenticatedUser(id=uuid4())

    with pytest.raises(FrozenInstanceError):
        user.email = "changed@example.com"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("user_id", "email"),
    [
        (UUID(int=0), None),
        (uuid4(), ""),
        (uuid4(), "   "),
    ],
)
def test_authenticated_user_rejects_invalid_identity(
    user_id: UUID,
    email: str | None,
) -> None:
    with pytest.raises(ValueError):
        AuthenticatedUser(id=user_id, email=email)


@pytest.mark.asyncio
async def test_access_token_verifier_port_accepts_infrastructure_adapter() -> None:
    user = AuthenticatedUser(id=uuid4())
    verifier = FakeAccessTokenVerifier(user)
    accepts_access_token_verifier(verifier)

    verified_user = await verifier.verify("access-token")

    assert verified_user == user
    assert verifier.tokens == ["access-token"]


@pytest.mark.parametrize(
    "error_type",
    [
        MissingAccessTokenError,
        InvalidAccessTokenError,
        MalformedAccessTokenError,
        ExpiredAccessTokenError,
    ],
)
def test_access_token_errors_share_authentication_base(
    error_type: type[AuthenticationError],
) -> None:
    assert isinstance(error_type("Authentication failed"), AuthenticationError)


@pytest.mark.parametrize(
    "error_type",
    [MalformedAccessTokenError, ExpiredAccessTokenError],
)
def test_specific_token_errors_are_invalid_token_errors(
    error_type: type[InvalidAccessTokenError],
) -> None:
    assert isinstance(error_type("Authentication failed"), InvalidAccessTokenError)
