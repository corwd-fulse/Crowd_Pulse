"""Versioned floorplans and manual job ROI mappings, without result rewrites."""
from alembic import op
import sqlalchemy as sa

revision = '20261117_07'
down_revision = '20261113_06'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('venue_revisions',
        sa.Column('venue_id', sa.String(80), primary_key=True),
        sa.Column('revision', sa.Integer(), primary_key=True),
        sa.Column('configuration', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint('revision > 0', name='ck_venue_revision_positive'))


def downgrade():
    raise RuntimeError('Destructive downgrade disabled; preserve venue history')
