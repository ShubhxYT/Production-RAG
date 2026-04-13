"""add query log table

Revision ID: a1b2c3d4e5f6
Revises: 4b2d9cd9cf26
Create Date: 2026-04-04 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str]] = '4b2d9cd9cf26'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create query_logs table."""
    op.create_table(
        'query_logs',
        sa.Column('id', sa.UUID(as_uuid=False), nullable=False),
        sa.Column('request_id', sa.String(), nullable=True),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('answer', sa.Text(), nullable=False),
        sa.Column('sources', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('prompt_variant', sa.String(), nullable=True),
        sa.Column('prompt_version', sa.String(), nullable=True),
        sa.Column('retrieval_top_k', sa.Integer(), nullable=True),
        sa.Column('retrieval_result_count', sa.Integer(), nullable=True),
        sa.Column('latency_ms', sa.Float(), nullable=True),
        sa.Column('retrieval_ms', sa.Float(), nullable=True),
        sa.Column('generation_ms', sa.Float(), nullable=True),
        sa.Column('prompt_tokens', sa.Integer(), nullable=True),
        sa.Column('completion_tokens', sa.Integer(), nullable=True),
        sa.Column('model', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_query_logs_request_id', 'query_logs', ['request_id'], unique=False)
    op.create_index('ix_query_logs_created_at', 'query_logs', ['created_at'], unique=False)


def downgrade() -> None:
    """Drop query_logs table."""
    op.drop_index('ix_query_logs_created_at', table_name='query_logs')
    op.drop_index('ix_query_logs_request_id', table_name='query_logs')
    op.drop_table('query_logs')
