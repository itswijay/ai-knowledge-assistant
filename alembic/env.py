import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

from app.core.config import get_database_settings
from app.infrastructure.database.models import DocumentChunkModel
from app.infrastructure.database.session import create_database_engine

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = DocumentChunkModel.metadata


def database_url() -> str:
    settings = get_database_settings()
    return str(settings.database_url.get_secret_value())


def run_migrations_offline() -> None:
    """Generate migration SQL without opening a database connection."""

    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def apply_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Apply migrations through an async SQLAlchemy connection."""

    engine = create_database_engine(get_database_settings())
    try:
        async with engine.connect() as connection:
            await connection.run_sync(apply_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
