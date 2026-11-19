"""Enforce video-to-venue isolation even for simultaneous registration requests."""
from alembic import op
import sqlalchemy as sa
revision = '20261117_10'
down_revision = '20261117_09'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('capture_video_venues',
        sa.Column('video_id', sa.Uuid(), sa.ForeignKey('videos.id', ondelete='RESTRICT'), primary_key=True),
        sa.Column('venue_id', sa.String(80), nullable=False),
        sa.Column('venue_revision', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['venue_id', 'venue_revision'],
            ['venue_revisions.venue_id', 'venue_revisions.revision'], ondelete='RESTRICT'))
    op.create_index('ix_capture_video_venues_venue_id', 'capture_video_venues', ['venue_id'])


def downgrade():
    raise RuntimeError('Destructive downgrade disabled; preserve video ownership')
