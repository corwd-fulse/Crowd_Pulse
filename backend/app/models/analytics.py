"""Capture times are nullable facts; processing times never stand in for them."""

from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger, CheckConstraint, DateTime, Float, ForeignKey, Index,
    Integer, Numeric, String, Text, UniqueConstraint, Uuid, func, text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Video(Base):
    __tablename__ = "videos"
    __table_args__ = (
        CheckConstraint("file_size >= 0", name="ck_videos_file_size"),
        CheckConstraint("duration_ms IS NULL OR duration_ms >= 0", name="ck_videos_duration"),
        CheckConstraint("status IN ('uploaded', 'unavailable')", name="ck_videos_status"),
        UniqueConstraint("saved_filename", name="uq_videos_saved_filename"),
        UniqueConstraint("storage_path", name="uq_videos_storage_path"),
        Index("ix_videos_uploaded_at", "uploaded_at"),
        Index("ix_videos_camera_id", "camera_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    camera_id: Mapped[int | None] = mapped_column(ForeignKey("cameras.id", ondelete="RESTRICT"))
    filename: Mapped[str] = mapped_column(Text)
    saved_filename: Mapped[str] = mapped_column(String(64))
    storage_path: Mapped[str] = mapped_column(Text, comment="Path relative to backend/, never a public URL")
    file_size: Mapped[int] = mapped_column(BigInteger)
    content_type: Mapped[str | None] = mapped_column(String(255))
    duration_ms: Mapped[int | None] = mapped_column(BigInteger)
    recorded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="Actual first-frame capture time; NULL if unknown")
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    status: Mapped[str] = mapped_column(String(20), server_default="uploaded")
    data_source: Mapped[str] = mapped_column(String(30), server_default="user_upload")



class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_analysis_jobs_status",
        ),
        CheckConstraint(
            "finished_at IS NULL OR (started_at IS NOT NULL AND finished_at >= started_at)",
            name="ck_analysis_jobs_times",
        ),
        Index(
            "ix_analysis_jobs_video_created",
            "video_id",
            "created_at",
        ),
        Index(
            "ix_analysis_jobs_status",
            "status",
        ),
        # 같은 영상에 진행 중인 분석 작업은 하나만 허용
        Index(
            "uq_analysis_jobs_active_video",
            "video_id",
            unique=True,
            postgresql_where=text(
                "status IN ('queued', 'running')"
            ),
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid4
    )
    video_id: Mapped[UUID] = mapped_column(
        ForeignKey("videos.id", ondelete="RESTRICT")
    )
    status: Mapped[str] = mapped_column(
        String(20), server_default="queued"
    )
    model_version: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    error_message: Mapped[str | None] = mapped_column(Text)


class CrowdMetric(Base):
    __tablename__ = "crowd_metrics"
    __table_args__ = (
        CheckConstraint("video_offset_ms >= 0", name="ck_crowd_metrics_offset"),
        CheckConstraint("person_count >= 0", name="ck_crowd_metrics_count"),
        CheckConstraint("congestion_percent BETWEEN 0 AND 100", name="ck_crowd_metrics_congestion"),
        CheckConstraint("density IS NULL OR density >= 0", name="ck_crowd_metrics_density"),
        Index("ix_crowd_metrics_analyzed_at", "analyzed_at"),
        Index("ix_crowd_metrics_job_offset", "analysis_job_id", "region_key", "video_offset_ms"),
        Index("ix_crowd_metrics_job_time", "analysis_job_id", "analyzed_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    analyzed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, nullable=False, comment="Actual analysis time; hypertable partition key; explicitly supplied by worker")
    analysis_job_id: Mapped[UUID] = mapped_column(ForeignKey("analysis_jobs.id", ondelete="RESTRICT"))
    region_key: Mapped[str] = mapped_column(String(100), server_default="frame")
    video_offset_ms: Mapped[int] = mapped_column(BigInteger)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="Actual frame capture time; NULL if unknown")
    person_count: Mapped[int] = mapped_column(Integer)
    congestion_percent: Mapped[Decimal] = mapped_column(Numeric(5, 2))
    density: Mapped[float | None] = mapped_column(Float, comment="People per square metre; NULL without calibrated area")


class CrowdPrediction(Base):
    __tablename__ = "crowd_predictions"
    __table_args__ = (
        CheckConstraint("source_offset_ms >= 0 AND target_offset_ms > source_offset_ms", name="ck_crowd_predictions_offsets"),
        CheckConstraint("predicted_person_count >= 0", name="ck_crowd_predictions_count"),
        CheckConstraint("predicted_congestion_percent BETWEEN 0 AND 100", name="ck_crowd_predictions_congestion"),
        CheckConstraint("confidence IS NULL OR confidence BETWEEN 0 AND 1", name="ck_crowd_predictions_confidence"),
        Index("ix_crowd_predictions_predicted_at", "predicted_at"),
        Index("ix_crowd_predictions_job_target", "analysis_job_id", "region_key", "target_offset_ms"),
        Index("ix_crowd_predictions_job_time", "analysis_job_id", "predicted_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, nullable=False, comment="Actual prediction generation time; hypertable partition key")
    analysis_job_id: Mapped[UUID] = mapped_column(ForeignKey("analysis_jobs.id", ondelete="RESTRICT"))
    region_key: Mapped[str] = mapped_column(String(100), server_default="frame")
    source_offset_ms: Mapped[int] = mapped_column(BigInteger)
    target_offset_ms: Mapped[int] = mapped_column(BigInteger)
    target_captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="Capture time corresponding to target offset, only when known")
    predicted_person_count: Mapped[int] = mapped_column(Integer)
    predicted_congestion_percent: Mapped[Decimal] = mapped_column(Numeric(5, 2))
    confidence: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(100))


class BottleneckEvent(Base):
    __tablename__ = "bottleneck_events"
    __table_args__ = (
        CheckConstraint("start_offset_ms >= 0 AND (end_offset_ms IS NULL OR end_offset_ms >= start_offset_ms)", name="ck_bottleneck_events_offsets"),
        CheckConstraint("captured_end_at IS NULL OR (captured_start_at IS NOT NULL AND captured_end_at >= captured_start_at)", name="ck_bottleneck_events_capture_times"),
        CheckConstraint("severity IN ('low', 'medium', 'high', 'critical')", name="ck_bottleneck_events_severity"),
        CheckConstraint("peak_person_count IS NULL OR peak_person_count >= 0", name="ck_bottleneck_events_count"),
        Index("ix_bottleneck_events_job_offset", "analysis_job_id", "start_offset_ms"),
        Index("ix_bottleneck_events_detected_at", "detected_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    analysis_job_id: Mapped[UUID] = mapped_column(ForeignKey("analysis_jobs.id", ondelete="RESTRICT"))
    region_key: Mapped[str] = mapped_column(String(100), server_default="frame")
    start_offset_ms: Mapped[int] = mapped_column(BigInteger)
    end_offset_ms: Mapped[int | None] = mapped_column(BigInteger)
    captured_start_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    captured_end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    severity: Mapped[str] = mapped_column(String(20))
    peak_person_count: Mapped[int | None] = mapped_column(Integer)
    description: Mapped[str | None] = mapped_column(Text)
