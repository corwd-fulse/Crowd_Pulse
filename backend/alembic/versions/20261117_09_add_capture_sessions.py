"""Append-only capture sessions, pinned to immutable venue revisions."""
from alembic import op
import sqlalchemy as sa
revision = '20261117_09'
down_revision = '20261117_08'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('capture_session_revisions',
        sa.Column('session_id', sa.Uuid(), primary_key=True),
        sa.Column('revision', sa.Integer(), primary_key=True),
        sa.Column('venue_id', sa.String(80), nullable=False),
        sa.Column('venue_revision', sa.Integer(), nullable=False),
        sa.Column('data', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['venue_id', 'venue_revision'],
            ['venue_revisions.venue_id', 'venue_revisions.revision'], ondelete='RESTRICT'))
    op.create_index('ix_capture_session_revisions_venue_id', 'capture_session_revisions', ['venue_id'])


def downgrade():
    raise RuntimeError('Destructive downgrade disabled; preserve capture evidence')
