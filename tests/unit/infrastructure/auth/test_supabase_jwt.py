from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from json import JSONDecodeError
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from jwt import PyJWK
from jwt.exceptions import (
    PyJWKClientConnectionError,
    PyJWKClientError,
    PyJWKError,
    PyJWKSetError,
)

from app.domain.errors import (
    AccessTokenVerificationError,
    ExpiredAccessTokenError,
    InvalidAccessTokenError,
    MalformedAccessTokenError,
)
from app.infrastructure.auth.supabase_jwt import (
    JWKS_REQUEST_TIMEOUT_SECONDS,
    SUPPORTED_JWT_ALGORITHMS,
    SupabaseJWTVerifier,
)

ISSUER = "https://test-project.supabase.co/auth/v1"
JWKS_URL = f"{ISSUER}/.well-known/jwks.json"
AUDIENCE = "authenticated"
KEY_ID = "test-key"


@dataclass
class FakeSigningKeyClient:
    signing_key: PyJWK
    error: Exception | None = None
    tokens: list[str | bytes] = field(default_factory=list)

    async def get_signing_key_from_jwt(self, token: str | bytes) -> PyJWK:
        self.tokens.append(token)
        if self.error is not None:
            raise self.error
        return self.signing_key


@pytest.fixture
def private_key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


@pytest.fixture
def signing_key(private_key: ec.EllipticCurvePrivateKey) -> PyJWK:
    key_data = jwt.algorithms.ECAlgorithm.to_jwk(
        private_key.public_key(),
        as_dict=True,
    )
    key_data.update({"kid": KEY_ID, "alg": "ES256", "use": "sig"})
    return PyJWK.from_dict(key_data)


def build_claims(**overrides: Any) -> dict[str, Any]:
    now = datetime.now(UTC)
    claims: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": now + timedelta(minutes=5),
        "iat": now,
        "sub": str(uuid4()),
        "role": "authenticated",
        "is_anonymous": False,
        "email": "user@example.com",
    }
    claims.update(overrides)
    return claims


def encode_token(
    private_key: ec.EllipticCurvePrivateKey,
    claims: dict[str, Any] | None = None,
) -> str:
    return jwt.encode(
        claims or build_claims(),
        private_key,
        algorithm="ES256",
        headers={"kid": KEY_ID},
    )


def build_verifier(client: FakeSigningKeyClient) -> SupabaseJWTVerifier:
    return SupabaseJWTVerifier(
        jwks_url=JWKS_URL,
        issuer=ISSUER,
        audience=AUDIENCE,
        cache_ttl_seconds=600,
        jwk_client=client,
    )


@pytest.mark.asyncio
async def test_valid_token_returns_only_application_identity(
    private_key: ec.EllipticCurvePrivateKey,
    signing_key: PyJWK,
) -> None:
    user_id = uuid4()
    token = encode_token(private_key, build_claims(sub=str(user_id)))
    client = FakeSigningKeyClient(signing_key)

    user = await build_verifier(client).verify(token)

    assert user.id == user_id
    assert user.email == "user@example.com"
    assert client.tokens == [token]


@pytest.mark.asyncio
async def test_email_is_optional_and_whitespace_is_not_exposed(
    private_key: ec.EllipticCurvePrivateKey,
    signing_key: PyJWK,
) -> None:
    token = encode_token(private_key, build_claims(email="   "))

    user = await build_verifier(FakeSigningKeyClient(signing_key)).verify(token)

    assert user.email is None


@pytest.mark.asyncio
async def test_expired_token_has_specific_error(
    private_key: ec.EllipticCurvePrivateKey,
    signing_key: PyJWK,
) -> None:
    token = encode_token(
        private_key,
        build_claims(exp=datetime.now(UTC) - timedelta(seconds=1)),
    )

    with pytest.raises(ExpiredAccessTokenError, match="expired"):
        await build_verifier(FakeSigningKeyClient(signing_key)).verify(token)


@pytest.mark.asyncio
async def test_invalid_signature_is_rejected(
    private_key: ec.EllipticCurvePrivateKey,
    signing_key: PyJWK,
) -> None:
    other_key = ec.generate_private_key(ec.SECP256R1())
    token = encode_token(other_key)

    with pytest.raises(InvalidAccessTokenError, match="invalid"):
        await build_verifier(FakeSigningKeyClient(signing_key)).verify(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["", "   ", "not-a-jwt", "one.two.three"])
async def test_malformed_token_is_rejected(
    token: str,
    signing_key: PyJWK,
) -> None:
    with pytest.raises(MalformedAccessTokenError, match="malformed"):
        await build_verifier(FakeSigningKeyClient(signing_key)).verify(token)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claims",
    [
        build_claims(iss="https://attacker.example/auth/v1"),
        build_claims(aud="anon"),
        build_claims(role="service_role"),
        build_claims(is_anonymous=True),
        build_claims(sub="not-a-uuid"),
        build_claims(sub=str(UUID(int=0))),
        build_claims(email=123),
    ],
)
async def test_untrusted_identity_claims_are_rejected(
    claims: dict[str, Any],
    private_key: ec.EllipticCurvePrivateKey,
    signing_key: PyJWK,
) -> None:
    token = encode_token(private_key, claims)

    with pytest.raises(InvalidAccessTokenError, match="invalid"):
        await build_verifier(FakeSigningKeyClient(signing_key)).verify(token)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_claim",
    ["iss", "aud", "sub", "exp", "iat", "role", "is_anonymous"],
)
async def test_required_claims_are_enforced(
    missing_claim: str,
    private_key: ec.EllipticCurvePrivateKey,
    signing_key: PyJWK,
) -> None:
    claims = build_claims()
    del claims[missing_claim]
    token = encode_token(private_key, claims)

    with pytest.raises(InvalidAccessTokenError, match="invalid"):
        await build_verifier(FakeSigningKeyClient(signing_key)).verify(token)


@pytest.mark.asyncio
async def test_shared_secret_algorithm_is_not_accepted(signing_key: PyJWK) -> None:
    token = jwt.encode(
        build_claims(),
        "not-a-real-supabase-secret-with-at-least-32-bytes",
        algorithm="HS256",
        headers={"kid": KEY_ID},
    )

    with pytest.raises(InvalidAccessTokenError, match="invalid"):
        await build_verifier(FakeSigningKeyClient(signing_key)).verify(token)


@pytest.mark.asyncio
async def test_unknown_signing_key_is_an_invalid_token(signing_key: PyJWK) -> None:
    client = FakeSigningKeyClient(
        signing_key,
        error=PyJWKClientError("unknown key identifier: sensitive-key-id"),
    )

    with pytest.raises(InvalidAccessTokenError) as error:
        await build_verifier(client).verify("header.payload.signature")

    assert "sensitive-key-id" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_error",
    [
        PyJWKClientConnectionError("connection details"),
        PyJWKSetError("invalid provider key set"),
        PyJWKError("invalid provider key"),
        JSONDecodeError("invalid provider response", "sensitive response", 0),
    ],
)
async def test_jwks_provider_failure_is_not_reported_as_invalid_credentials(
    client_error: Exception,
    signing_key: PyJWK,
) -> None:
    client = FakeSigningKeyClient(signing_key, error=client_error)

    with pytest.raises(AccessTokenVerificationError) as error:
        await build_verifier(client).verify("header.payload.signature")

    assert "connection details" not in str(error.value)
    assert "provider key set" not in str(error.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("jwks_url", ""),
        ("issuer", "   "),
        ("audience", ""),
        ("cache_ttl_seconds", 59),
        ("cache_ttl_seconds", 3601),
    ],
)
def test_verifier_rejects_invalid_configuration(
    field: str,
    value: str | int,
    signing_key: PyJWK,
) -> None:
    arguments: dict[str, Any] = {
        "jwks_url": JWKS_URL,
        "issuer": ISSUER,
        "audience": AUDIENCE,
        "cache_ttl_seconds": 600,
        "jwk_client": FakeSigningKeyClient(signing_key),
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        SupabaseJWTVerifier(**arguments)


def test_default_jwks_client_honors_bounded_key_set_cache() -> None:
    with patch(
        "app.infrastructure.auth.supabase_jwt.PyJWKClient",
        autospec=True,
    ) as client_type:
        SupabaseJWTVerifier(
            jwks_url=JWKS_URL,
            issuer=ISSUER,
            audience=AUDIENCE,
            cache_ttl_seconds=900,
        )

    client_type.assert_called_once_with(
        JWKS_URL,
        cache_keys=False,
        cache_jwk_set=True,
        lifespan=900.0,
        timeout=JWKS_REQUEST_TIMEOUT_SECONDS,
    )
    assert set(SUPPORTED_JWT_ALGORITHMS) == {"ES256", "RS256", "EdDSA"}
