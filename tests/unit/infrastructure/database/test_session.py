import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import DatabaseSettings
from app.infrastructure.database.session import (
    create_database_engine,
    create_session_factory,
)


def database_settings() -> DatabaseSettings:
    return DatabaseSettings(
        _env_file=None,
        database_url="postgresql+asyncpg://postgres:password@localhost/app",
    )


@pytest.mark.asyncio
async def test_async_database_engine_masks_password() -> None:
    engine = create_database_engine(database_settings())

    try:
        assert engine.dialect.name == "postgresql"
        assert engine.dialect.driver == "asyncpg"
        assert "password" not in str(engine.url)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_session_factory_creates_non_expiring_async_sessions() -> None:
    engine = create_database_engine(database_settings())
    session_factory = create_session_factory(engine)

    try:
        async with session_factory() as session:
            assert isinstance(session, AsyncSession)
            assert session.sync_session.expire_on_commit is False
    finally:
        await engine.dispose()
