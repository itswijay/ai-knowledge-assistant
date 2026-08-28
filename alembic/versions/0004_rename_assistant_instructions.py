"""Rename assistant system prompt to assistant instructions.

Revision ID: 0004
Revises: 0003
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "assistants",
        "system_prompt",
        new_column_name="assistant_instructions",
    )
    op.execute(
        "ALTER TABLE assistants "
        "RENAME CONSTRAINT ck_assistants_system_prompt_not_blank "
        "TO ck_assistants_assistant_instructions_not_blank"
    )


def downgrade() -> None:
    op.alter_column(
        "assistants",
        "assistant_instructions",
        new_column_name="system_prompt",
    )
    op.execute(
        "ALTER TABLE assistants "
        "RENAME CONSTRAINT ck_assistants_assistant_instructions_not_blank "
        "TO ck_assistants_system_prompt_not_blank"
    )
