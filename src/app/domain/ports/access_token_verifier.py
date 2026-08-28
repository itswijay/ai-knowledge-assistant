from typing import Protocol

from app.domain.entities.authentication import AuthenticatedUser


class AccessTokenVerifier(Protocol):
    async def verify(self, token: str) -> AuthenticatedUser: ...
