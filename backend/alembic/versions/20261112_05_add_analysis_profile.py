"""Distinguish fast/full requests without changing existing job states."""
from alembic import op
import sqlalchemy as sa

revision = '20261112_05'
down_revision = '20261112_04'
branch_labels = None
depends_on = None

def upgrade():
    op.add_column('analysis_jobs', sa.Column('analysis_profile', sa.String(10), nullable=False, server_default='full'))
    op.create_check_constraint('ck_analysis_jobs_profile', 'analysis_jobs', "analysis_profile IN ('fast', 'full')")

def downgrade():
    raise RuntimeError('Destructive downgrade disabled; preserve analysis profile history')
