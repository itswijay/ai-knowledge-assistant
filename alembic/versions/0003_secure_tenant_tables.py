"""Deny direct Data API access to tenant tables.

Revision ID: 0003
Revises: 0002
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

PROTECTED_TABLES = (
    "organizations",
    "organization_members",
    "assistants",
    "documents",
    "document_chunks",
)


def upgrade() -> None:
    for table_name in PROTECTED_TABLES:
        op.execute(f"ALTER TABLE public.{table_name} ENABLE ROW LEVEL SECURITY")
        op.execute(f"REVOKE ALL ON TABLE public.{table_name} FROM anon, authenticated")


def downgrade() -> None:
    # This restores the standard Supabase client-role privileges that this
    # revision removes. Downgrading therefore intentionally weakens isolation.
    for table_name in reversed(PROTECTED_TABLES):
        op.execute(f"ALTER TABLE public.{table_name} DISABLE ROW LEVEL SECURITY")
        op.execute(
            "GRANT SELECT, INSERT, UPDATE, DELETE "
            f"ON TABLE public.{table_name} TO anon, authenticated"
        )
