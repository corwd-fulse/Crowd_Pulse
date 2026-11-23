"""Store DM-Count density-map estimates as a supplementary reference metric (FR-DEN-01/02, P2).

Revision ID: 20261122_07
Revises: 20261120_12

Same JSONB-bundle pattern as the rest of vision_extended_results.py (dwell_times,
queue_scores, etc.) — DM-Count is a completely independent model from YOLO+ByteTrack
(shares only the input video frames) so it gets a column on the existing extension table
rather than a new one, and it is nullable/optional like crowd_speed_zone because it is never
required for a job to succeed and most jobs will not run it (slow CPU inference, see
density_estimation.py). Each record already carries model_version/inference_ms inline
(same shape as the other bundle columns), so no separate relational columns are added here.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '20261122_07'
down_revision = '20261120_12'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('vision_extended_results',
        sa.Column('dmcount_estimates', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade():
    raise RuntimeError('Destructive downgrade disabled; preserve results with a forward migration')
