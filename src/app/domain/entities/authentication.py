from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    """Verified user identity exposed to application code."""

    id: UUID
    email: str | None = None

    def __post_init__(self) -> None:
        if self.id.int == 0:
            raise ValueError("id must not be the nil UUID")
        if self.email is not None and not self.email.strip():
            raise ValueError("email must not be blank when provided")
