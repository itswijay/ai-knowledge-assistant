from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import get_authenticated_user
from app.domain.entities import AuthenticatedUser
from app.presentation.api.schemas.authentication import CurrentUserResponse

router = APIRouter(prefix="/api/v1", tags=["authentication"])


@router.get("/me", response_model=CurrentUserResponse)
async def get_current_user(
    user: Annotated[AuthenticatedUser, Depends(get_authenticated_user)],
) -> CurrentUserResponse:
    return CurrentUserResponse(id=user.id, email=user.email)
