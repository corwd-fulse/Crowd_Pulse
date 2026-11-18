"""Store supplementary Vision AI metrics (heatmap/dwell/counter-flow/queue/tracking quality).

Revision ID: 20261112_06
Revises: 20261112_05

One JSONB-based table, not five fully normalized ones like vision_zone_metrics /
vision_bottleneck_scores. These metrics are additive extensions beyond the original
FR-ZON spec (dwell time, counter flow, queue analysis, density heatmap, tracking quality,
calibrated crowd speed) and are never required for a job to succeed, so they get a lighter
schema than the core five-file result contract in vision.py.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '20261112_06'
down_revision = '20261112_05'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('vision_extended_results',
        sa.Column('analysis_job_id', sa.Uuid(), nullable=False),
        sa.Column('imported_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('pixels_per_meter', sa.Float(), nullable=True),
        sa.Column('artifacts', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('dwell_times', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('counter_flow', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('queue_scores', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('density_heatmap', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('tracking_quality', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('crowd_speed_zone', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('crowd_speed_individual', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint('pixels_per_meter IS NULL OR pixels_per_meter > 0', name='ck_vision_extended_calibration'),
        sa.CheckConstraint(
            '(crowd_speed_zone IS NULL) = (pixels_per_meter IS NULL)',
            name='ck_vision_extended_speed_requires_calibration',
        ),
        sa.ForeignKeyConstraint(['analysis_job_id'], ['vision_results.analysis_job_id'], ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('analysis_job_id'),
    )


def downgrade():
    raise RuntimeError('Destructive downgrade disabled; preserve results with a forward migration')
