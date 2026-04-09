"""add feedback_logs table

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-04-06 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str]] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create feedback_logs table."""
    op.create_table(
        'feedback_logs',
        sa.Column('id', sa.UUID(as_uuid=False), nullable=False),
        sa.Column('query_log_id', sa.String(), nullable=True),
        sa.Column('feedback_type', sa.String(), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=True),
        sa.Column('correction', sa.Text(), nullable=True),
        sa.Column('query_text', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_feedback_logs_query_log_id', 'feedback_logs', ['query_log_id'], unique=False)
    op.create_index('ix_feedback_logs_created_at', 'feedback_logs', ['created_at'], unique=False)


def downgrade() -> None:
    """Drop feedback_logs table."""
    op.drop_index('ix_feedback_logs_created_at', table_name='feedback_logs')
    op.drop_index('ix_feedback_logs_query_log_id', table_name='feedback_logs')
    op.drop_table('feedback_logs')
