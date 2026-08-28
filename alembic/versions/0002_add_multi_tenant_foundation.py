"""Add organizations, memberships, assistants, and document ownership.

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(btrim(name)) > 0",
            name="name_not_blank",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_organizations"),
    )
    op.create_table(
        "organization_members",
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "role IN ('owner', 'admin', 'member')",
            name="role_valid",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organizations.id"],
            name="fk_organization_members_organization_id_organizations",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["auth.users.id"],
            name="fk_organization_members_user_id_auth_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "organization_id",
            "user_id",
            name="pk_organization_members",
        ),
    )
    op.create_index(
        "ix_organization_members_organization_id",
        "organization_members",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ix_organization_members_user_id",
        "organization_members",
        ["user_id"],
        unique=False,
    )
    op.create_table(
        "assistants",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.String(length=1000), nullable=True),
        sa.Column(
            "welcome_message",
            sa.String(length=500),
            server_default=sa.text("'Hi! How can I help you today?'"),
            nullable=False,
        ),
        sa.Column(
            "system_prompt",
            sa.String(length=4000),
            server_default=sa.text(
                "'Answer questions using the provided knowledge base.'"
            ),
            nullable=False,
        ),
        sa.Column("logo_url", sa.String(length=2048), nullable=True),
        sa.Column(
            "primary_color",
            sa.String(length=7),
            server_default=sa.text("'#2563EB'"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(btrim(name)) > 0",
            name="name_not_blank",
        ),
        sa.CheckConstraint(
            "description IS NULL OR length(btrim(description)) > 0",
            name="description_not_blank",
        ),
        sa.CheckConstraint(
            "length(btrim(welcome_message)) > 0",
            name="welcome_message_not_blank",
        ),
        sa.CheckConstraint(
            "length(btrim(system_prompt)) > 0",
            name="system_prompt_not_blank",
        ),
        sa.CheckConstraint(
            "logo_url IS NULL OR length(btrim(logo_url)) > 0",
            name="logo_url_not_blank",
        ),
        sa.CheckConstraint(
            "primary_color ~ '^#[0-9A-Fa-f]{6}$'",
            name="primary_color_valid",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organizations.id"],
            name="fk_assistants_organization_id_organizations",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_assistants"),
    )
    op.create_index(
        "ix_assistants_organization_id",
        "assistants",
        ["organization_id"],
        unique=False,
    )

    # Phase 1 rows are development/test data with no tenant owner. Removing them
    # avoids inventing an inaccessible legacy organization or nullable ownership.
    op.execute("DELETE FROM documents")
    op.add_column(
        "documents",
        sa.Column(
            "assistant_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
    )
    op.create_foreign_key(
        "fk_documents_assistant_id_assistants",
        "documents",
        "assistants",
        ["assistant_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_documents_assistant_id",
        "documents",
        ["assistant_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_documents_assistant_id", table_name="documents")
    op.drop_constraint(
        "fk_documents_assistant_id_assistants",
        "documents",
        type_="foreignkey",
    )
    op.drop_column("documents", "assistant_id")
    op.drop_index("ix_assistants_organization_id", table_name="assistants")
    op.drop_table("assistants")
    op.drop_index(
        "ix_organization_members_user_id",
        table_name="organization_members",
    )
    op.drop_index(
        "ix_organization_members_organization_id",
        table_name="organization_members",
    )
    op.drop_table("organization_members")
    op.drop_table("organizations")
