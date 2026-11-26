"""Persist video ROI time-series forecasts in the database.

Revision ID: 20261126_08
Revises: 20261122_07

Until now forecast/run_job_forecast.py results lived only as files under
uploads/results/<job>/forecast/<source>/ on one PC. This stores them in the DB:

- crowd_predictions (existing hypertable, never written before) gets the columns a
  video forecast point needs: count source (yolo/dmcount/fused), model name, horizon,
  backtest/forecast split, exact float prediction, observed value and status.
  predicted_congestion_percent becomes nullable: ROI area is an assumed value
  (50 px/m), so no congestion percent is written rather than an invented one.
  Rows keep analysis_job_id/region_key, so they cannot mix with the Seoul
  administrative-dong population forecast (served by its own API, not stored here).
- video_forecast_runs holds one row per (job, count source): summary (backtest
  metrics, minute-level results), anomaly level segments/events, and provenance.
  Unique (analysis_job_id, count_source) makes re-imports replace instead of duplicate.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '20261126_08'
down_revision = '20261122_07'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('crowd_predictions', sa.Column('count_source', sa.String(20), nullable=True,
                  comment='Video forecast count source: yolo | dmcount | fused'))
    op.add_column('crowd_predictions', sa.Column('model_name', sa.String(50), nullable=True))
    op.add_column('crowd_predictions', sa.Column('horizon_s', sa.Integer, nullable=True))
    op.add_column('crowd_predictions', sa.Column('split', sa.String(10), nullable=True,
                  comment='backtest (actual known) | forecast (after video end)'))
    op.add_column('crowd_predictions', sa.Column('predicted_value', sa.Float, nullable=True,
                  comment='Exact predicted count (predicted_person_count is its rounded integer)'))
    op.add_column('crowd_predictions', sa.Column('actual_count', sa.Float, nullable=True))
    op.add_column('crowd_predictions', sa.Column('status', sa.String(40), nullable=True))
    op.alter_column('crowd_predictions', 'predicted_congestion_percent', existing_type=sa.Numeric(5, 2), nullable=True)
    op.create_check_constraint('ck_crowd_predictions_split', 'crowd_predictions',
                               "split IS NULL OR split IN ('backtest', 'forecast')")
    op.create_index('ix_crowd_predictions_job_source_series', 'crowd_predictions',
                    ['analysis_job_id', 'count_source', 'horizon_s', 'region_key', 'target_offset_ms'])

    op.create_table(
        'video_forecast_runs',
        sa.Column('id', sa.Uuid, primary_key=True),
        sa.Column('analysis_job_id', sa.Uuid, sa.ForeignKey('analysis_jobs.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('count_source', sa.String(20), nullable=False),
        sa.Column('predicted_at', sa.DateTime(timezone=True), nullable=False,
                  comment='Same value as the crowd_predictions rows of this run'),
        sa.Column('imported_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('summary', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('level_segments', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('level_events', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('provenance', postgresql.JSONB(astext_type=sa.Text()), nullable=True,
                  comment='Where the run came from (result folder, pretrained model dir/version)'),
        sa.Column('point_count', sa.Integer, nullable=False),
        sa.CheckConstraint("count_source IN ('yolo', 'dmcount', 'fused')", name='ck_video_forecast_runs_source'),
        sa.UniqueConstraint('analysis_job_id', 'count_source', name='uq_video_forecast_runs_job_source'),
    )


def downgrade():
    raise RuntimeError('Destructive downgrade disabled; preserve results with a forward migration')
