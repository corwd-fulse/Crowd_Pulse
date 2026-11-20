"""Expose live Optical Flow progress/ETA for long-running FULL analysis jobs.

Additive/nullable only: existing rows and the fast/running/succeeded lifecycle are
untouched. Nothing backfills historical jobs; only a running job's worker updates these
going forward (see app/worker.py's progress_callback wiring).
"""
from alembic import op
import sqlalchemy as sa

revision = '20261120_12'
down_revision = '20261119_11'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('analysis_jobs', sa.Column('progress_percent', sa.Float(), nullable=True))
    op.add_column('analysis_jobs', sa.Column('progress_eta_seconds', sa.Float(), nullable=True))
    op.add_column('analysis_jobs', sa.Column('progress_updated_at', sa.DateTime(timezone=True), nullable=True))


def downgrade():
    raise RuntimeError('Destructive downgrade disabled; preserve progress history')
