"""Add video analysis schema and Timescale hypertables

Revision ID: 20261112_01
Revises:
"""
from alembic import op
import sqlalchemy as sa


revision = '20261112_01'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Only inspect the pre-installed extension; never install or configure it.
    op.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
                RAISE EXCEPTION 'The existing timescaledb extension is required';
            END IF;
        END $$;
    """)
    # CREATE TABLE deliberately fails on name collisions. Never adopt, drop or
    # silently modify an existing table. PostgreSQL rolls back this entire DDL.
    op.create_table('cameras',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(length=100), nullable=False),
    sa.Column('location', sa.String(length=255), nullable=True),
    sa.Column('stream_url', sa.Text(), nullable=True),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('videos',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('camera_id', sa.Integer(), nullable=True),
    sa.Column('filename', sa.Text(), nullable=False),
    sa.Column('saved_filename', sa.String(length=64), nullable=False),
    sa.Column('storage_path', sa.Text(), nullable=False, comment='Path relative to backend/, never a public URL'),
    sa.Column('file_size', sa.BigInteger(), nullable=False),
    sa.Column('content_type', sa.String(length=255), nullable=True),
    sa.Column('duration_ms', sa.BigInteger(), nullable=True),
    sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=True, comment='Actual first-frame capture time; NULL if unknown'),
    sa.Column('uploaded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('status', sa.String(length=20), server_default='uploaded', nullable=False),
    sa.Column('data_source', sa.String(length=30), server_default='user_upload', nullable=False),
    sa.CheckConstraint("status IN ('uploaded', 'unavailable')", name='ck_videos_status'),
    sa.CheckConstraint('duration_ms IS NULL OR duration_ms >= 0', name='ck_videos_duration'),
    sa.CheckConstraint('file_size >= 0', name='ck_videos_file_size'),
    sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('saved_filename', name='uq_videos_saved_filename'),
    sa.UniqueConstraint('storage_path', name='uq_videos_storage_path')
    )
    op.create_index('ix_videos_camera_id', 'videos', ['camera_id'], unique=False)
    op.create_index('ix_videos_uploaded_at', 'videos', ['uploaded_at'], unique=False)
    op.create_table('analysis_jobs',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('video_id', sa.Uuid(), nullable=False),
    sa.Column('status', sa.String(length=20), server_default='queued', nullable=False),
    sa.Column('model_version', sa.String(length=100), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.CheckConstraint("status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')", name='ck_analysis_jobs_status'),
    sa.CheckConstraint('finished_at IS NULL OR (started_at IS NOT NULL AND finished_at >= started_at)', name='ck_analysis_jobs_times'),
    sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_analysis_jobs_status', 'analysis_jobs', ['status'], unique=False)
    op.create_index('ix_analysis_jobs_video_created', 'analysis_jobs', ['video_id', 'created_at'], unique=False)
    op.create_table('bottleneck_events',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('analysis_job_id', sa.Uuid(), nullable=False),
    sa.Column('region_key', sa.String(length=100), server_default='frame', nullable=False),
    sa.Column('start_offset_ms', sa.BigInteger(), nullable=False),
    sa.Column('end_offset_ms', sa.BigInteger(), nullable=True),
    sa.Column('captured_start_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('captured_end_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('detected_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('severity', sa.String(length=20), nullable=False),
    sa.Column('peak_person_count', sa.Integer(), nullable=True),
    sa.Column('description', sa.Text(), nullable=True),
    sa.CheckConstraint("severity IN ('low', 'medium', 'high', 'critical')", name='ck_bottleneck_events_severity'),
    sa.CheckConstraint('captured_end_at IS NULL OR (captured_start_at IS NOT NULL AND captured_end_at >= captured_start_at)', name='ck_bottleneck_events_capture_times'),
    sa.CheckConstraint('peak_person_count IS NULL OR peak_person_count >= 0', name='ck_bottleneck_events_count'),
    sa.CheckConstraint('start_offset_ms >= 0 AND (end_offset_ms IS NULL OR end_offset_ms >= start_offset_ms)', name='ck_bottleneck_events_offsets'),
    sa.ForeignKeyConstraint(['analysis_job_id'], ['analysis_jobs.id'], ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_bottleneck_events_detected_at', 'bottleneck_events', ['detected_at'], unique=False)
    op.create_index('ix_bottleneck_events_job_offset', 'bottleneck_events', ['analysis_job_id', 'start_offset_ms'], unique=False)
    op.create_table('crowd_metrics',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('analyzed_at', sa.DateTime(timezone=True), nullable=False, comment='Actual analysis time; hypertable partition key; explicitly supplied by worker'),
    sa.Column('analysis_job_id', sa.Uuid(), nullable=False),
    sa.Column('region_key', sa.String(length=100), server_default='frame', nullable=False),
    sa.Column('video_offset_ms', sa.BigInteger(), nullable=False),
    sa.Column('captured_at', sa.DateTime(timezone=True), nullable=True, comment='Actual frame capture time; NULL if unknown'),
    sa.Column('person_count', sa.Integer(), nullable=False),
    sa.Column('congestion_percent', sa.Numeric(precision=5, scale=2), nullable=False),
    sa.Column('density', sa.Float(), nullable=True, comment='People per square metre; NULL without calibrated area'),
    sa.CheckConstraint('congestion_percent BETWEEN 0 AND 100', name='ck_crowd_metrics_congestion'),
    sa.CheckConstraint('density IS NULL OR density >= 0', name='ck_crowd_metrics_density'),
    sa.CheckConstraint('person_count >= 0', name='ck_crowd_metrics_count'),
    sa.CheckConstraint('video_offset_ms >= 0', name='ck_crowd_metrics_offset'),
    sa.ForeignKeyConstraint(['analysis_job_id'], ['analysis_jobs.id'], ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', 'analyzed_at')
    )
    op.create_index('ix_crowd_metrics_analyzed_at', 'crowd_metrics', ['analyzed_at'], unique=False)
    op.create_index('ix_crowd_metrics_job_offset', 'crowd_metrics', ['analysis_job_id', 'region_key', 'video_offset_ms'], unique=False)
    op.create_index('ix_crowd_metrics_job_time', 'crowd_metrics', ['analysis_job_id', 'analyzed_at'], unique=False)
    op.create_table('crowd_predictions',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('predicted_at', sa.DateTime(timezone=True), nullable=False, comment='Actual prediction generation time; hypertable partition key'),
    sa.Column('analysis_job_id', sa.Uuid(), nullable=False),
    sa.Column('region_key', sa.String(length=100), server_default='frame', nullable=False),
    sa.Column('source_offset_ms', sa.BigInteger(), nullable=False),
    sa.Column('target_offset_ms', sa.BigInteger(), nullable=False),
    sa.Column('target_captured_at', sa.DateTime(timezone=True), nullable=True, comment='Capture time corresponding to target offset, only when known'),
    sa.Column('predicted_person_count', sa.Integer(), nullable=False),
    sa.Column('predicted_congestion_percent', sa.Numeric(precision=5, scale=2), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('model_version', sa.String(length=100), nullable=False),
    sa.CheckConstraint('confidence IS NULL OR confidence BETWEEN 0 AND 1', name='ck_crowd_predictions_confidence'),
    sa.CheckConstraint('predicted_congestion_percent BETWEEN 0 AND 100', name='ck_crowd_predictions_congestion'),
    sa.CheckConstraint('predicted_person_count >= 0', name='ck_crowd_predictions_count'),
    sa.CheckConstraint('source_offset_ms >= 0 AND target_offset_ms > source_offset_ms', name='ck_crowd_predictions_offsets'),
    sa.ForeignKeyConstraint(['analysis_job_id'], ['analysis_jobs.id'], ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', 'predicted_at')
    )
    op.create_index('ix_crowd_predictions_job_target', 'crowd_predictions', ['analysis_job_id', 'region_key', 'target_offset_ms'], unique=False)
    op.create_index('ix_crowd_predictions_job_time', 'crowd_predictions', ['analysis_job_id', 'predicted_at'], unique=False)
    op.create_index('ix_crowd_predictions_predicted_at', 'crowd_predictions', ['predicted_at'], unique=False)
    op.execute("""
        SELECT create_hypertable(
            'crowd_metrics', by_range('analyzed_at', INTERVAL '1 day'),
            create_default_indexes => FALSE, migrate_data => FALSE
        )
    """)
    op.execute("""
        SELECT create_hypertable(
            'crowd_predictions', by_range('predicted_at', INTERVAL '1 day'),
            create_default_indexes => FALSE, migrate_data => FALSE
        )
    """)


def downgrade():
    raise RuntimeError(
        'Destructive downgrade is disabled to preserve data. Use a reviewed forward migration.'
    )
