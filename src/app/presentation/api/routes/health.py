from fastapi import APIRouter

from app.presentation.api.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report whether the API process is available."""

    return HealthResponse(status="ok")
