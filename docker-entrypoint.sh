#!/bin/sh
set -e

# Run database migrations if RUN_MIGRATIONS is set to "true" or "1"
if [ "$RUN_MIGRATIONS" = "true" ] || [ "$RUN_MIGRATIONS" = "1" ]; then
    echo "Running database migrations (alembic upgrade head)..."
    alembic upgrade head
    echo "Migrations applied successfully."
fi

# Execute the container command
exec "$@"
