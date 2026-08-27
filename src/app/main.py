from fastapi import FastAPI

from app.presentation.api.routes.health import router as health_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    application = FastAPI(
        title="AI Knowledge Assistant API",
        version="0.1.0",
    )
    application.include_router(health_router)
    return application


app = create_app()
