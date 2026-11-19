"""Append-only video ROI revisions; no legacy row or result rewrite."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
revision='20261113_06'
down_revision='20261112_05'
branch_labels=None
depends_on=None
def upgrade():
    op.create_table('video_roi_revisions',
        sa.Column('id',sa.Uuid(),primary_key=True),
        sa.Column('video_id',sa.Uuid(),sa.ForeignKey('videos.id',ondelete='RESTRICT'),nullable=False),
        sa.Column('ordinal',sa.Integer(),nullable=False),
        sa.Column('snapshot',postgresql.JSONB(),nullable=False),
        sa.Column('fingerprint',sa.String(64),nullable=False),
        sa.Column('created_at',sa.DateTime(timezone=True),server_default=sa.func.now(),nullable=False),
        sa.UniqueConstraint('video_id','ordinal',name='uq_video_roi_revision_ordinal'),
        sa.CheckConstraint('ordinal>0',name='ck_video_roi_revision_ordinal'))
    op.create_index('ix_video_roi_revisions_video_id','video_roi_revisions',['video_id'])
def downgrade():
    raise RuntimeError('Destructive downgrade disabled; preserve ROI history')
