from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import DatabaseSettings

type AsyncSessionFactory = async_sessionmaker[AsyncSession]


def create_database_engine(settings: DatabaseSettings) -> AsyncEngine:
    """Create an async engine without opening a database connection."""

    database_url = str(settings.database_url.get_secret_value())
    return create_async_engine(database_url, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> AsyncSessionFactory:
    """Create sessions that retain loaded state after a commit."""

    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
