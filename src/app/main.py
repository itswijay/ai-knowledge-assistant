from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.dependencies import build_application_container
from app.presentation.api.errors import register_exception_handlers
from app.presentation.api.routes.documents import router as documents_router
from app.presentation.api.routes.health import router as health_router


@asynccontextmanager
async def application_lifespan(application: FastAPI) -> AsyncIterator[None]:
    container = build_application_container(get_settings())
    application.state.container = container
    try:
        yield
    finally:
        await container.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    application = FastAPI(
        title="AI Knowledge Assistant API",
        version="0.1.0",
        lifespan=application_lifespan,
    )
    register_exception_handlers(application)
    application.include_router(health_router)
    application.include_router(documents_router)
    return application


app = create_app()
