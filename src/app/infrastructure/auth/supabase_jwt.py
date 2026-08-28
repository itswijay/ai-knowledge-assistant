import asyncio
from collections.abc import Mapping
from json import JSONDecodeError
from typing import Any, Protocol
from uuid import UUID

import jwt
from jwt import PyJWK, PyJWKClient
from jwt.exceptions import (
    DecodeError,
    ExpiredSignatureError,
    InvalidSignatureError,
    InvalidTokenError,
    PyJWKClientConnectionError,
    PyJWKClientError,
    PyJWKError,
    PyJWKSetError,
)

from app.domain.entities import AuthenticatedUser
from app.domain.errors import (
    AccessTokenVerificationError,
    ExpiredAccessTokenError,
    InvalidAccessTokenError,
    MalformedAccessTokenError,
)

SUPPORTED_JWT_ALGORITHMS = ("ES256", "RS256", "EdDSA")
REQUIRED_JWT_CLAIMS = (
    "iss",
    "aud",
    "exp",
    "iat",
    "sub",
    "role",
    "is_anonymous",
)
JWKS_REQUEST_TIMEOUT_SECONDS = 5.0


class AsyncSigningKeyClient(Protocol):
    async def get_signing_key_from_jwt(self, token: str | bytes) -> PyJWK: ...


class CachedPyJWKClient:
    """Async boundary around PyJWT's blocking, cached JWKS client."""

    def __init__(self, jwks_url: str, *, cache_ttl_seconds: int) -> None:
        self._client = PyJWKClient(
            jwks_url,
            cache_keys=False,
            cache_jwk_set=True,
            lifespan=float(cache_ttl_seconds),
            timeout=JWKS_REQUEST_TIMEOUT_SECONDS,
        )

    async def get_signing_key_from_jwt(self, token: str | bytes) -> PyJWK:
        return await asyncio.to_thread(self._client.get_signing_key_from_jwt, token)


class SupabaseJWTVerifier:
    """Verify Supabase access tokens locally with cached asymmetric keys."""

    def __init__(
        self,
        *,
        jwks_url: str,
        issuer: str,
        audience: str,
        cache_ttl_seconds: int,
        jwk_client: AsyncSigningKeyClient | None = None,
    ) -> None:
        if not jwks_url.strip():
            raise ValueError("jwks_url must not be blank")
        if not issuer.strip():
            raise ValueError("issuer must not be blank")
        if not audience.strip():
            raise ValueError("audience must not be blank")
        if not 60 <= cache_ttl_seconds <= 3600:
            raise ValueError("cache_ttl_seconds must be between 60 and 3600")

        self._issuer = issuer
        self._audience = audience
        self._jwk_client = jwk_client or CachedPyJWKClient(
            jwks_url,
            cache_ttl_seconds=cache_ttl_seconds,
        )

    async def verify(self, token: str) -> AuthenticatedUser:
        cleaned_token = token.strip()
        if not cleaned_token:
            raise MalformedAccessTokenError("Access token is malformed")

        signing_key = await self._resolve_signing_key(cleaned_token)
        claims = self._decode_claims(cleaned_token, signing_key)
        return self._to_authenticated_user(claims)

    async def _resolve_signing_key(self, token: str) -> PyJWK:
        try:
            return await self._jwk_client.get_signing_key_from_jwt(token)
        except PyJWKClientConnectionError as error:
            raise AccessTokenVerificationError(
                "Access token verification is temporarily unavailable"
            ) from error
        except DecodeError as error:
            raise MalformedAccessTokenError("Access token is malformed") from error
        except (
            JSONDecodeError,
            PyJWKError,
            PyJWKSetError,
            UnicodeDecodeError,
        ) as error:
            raise AccessTokenVerificationError(
                "Access token verification is temporarily unavailable"
            ) from error
        except PyJWKClientError as error:
            raise InvalidAccessTokenError("Access token is invalid") from error

    def _decode_claims(self, token: str, signing_key: PyJWK) -> Mapping[str, Any]:
        try:
            return jwt.decode(
                token,
                key=signing_key,
                algorithms=SUPPORTED_JWT_ALGORITHMS,
                audience=self._audience,
                issuer=self._issuer,
                options={"require": list(REQUIRED_JWT_CLAIMS)},
            )
        except ExpiredSignatureError as error:
            raise ExpiredAccessTokenError("Access token has expired") from error
        except InvalidSignatureError as error:
            raise InvalidAccessTokenError("Access token is invalid") from error
        except DecodeError as error:
            raise MalformedAccessTokenError("Access token is malformed") from error
        except InvalidTokenError as error:
            raise InvalidAccessTokenError("Access token is invalid") from error

    @staticmethod
    def _to_authenticated_user(claims: Mapping[str, Any]) -> AuthenticatedUser:
        if claims.get("role") != "authenticated":
            raise InvalidAccessTokenError("Access token is invalid")
        if claims.get("is_anonymous") is not False:
            raise InvalidAccessTokenError("Access token is invalid")

        try:
            user_id = UUID(claims["sub"])
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidAccessTokenError("Access token is invalid") from error

        email_claim = claims.get("email")
        if email_claim is not None and not isinstance(email_claim, str):
            raise InvalidAccessTokenError("Access token is invalid")
        email = email_claim.strip() if email_claim and email_claim.strip() else None

        try:
            return AuthenticatedUser(id=user_id, email=email)
        except ValueError as error:
            raise InvalidAccessTokenError("Access token is invalid") from error
