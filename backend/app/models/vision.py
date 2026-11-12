"""CSV observations are separate from legacy instantaneous counts/events."""
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import BigInteger, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, Numeric, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class VisionResult(Base):
    __tablename__ = "vision_results"
    __table_args__ = (
        CheckConstraint("width > 0 AND height > 0", name="ck_vision_results_resolution"),
        CheckConstraint("roi_status IN ('candidate', 'approved')", name="ck_vision_results_roi_status"),
        CheckConstraint("analysis_finished_at >= analysis_started_at", name="ck_vision_results_times"),
    )
    # Video is linked through analysis_jobs.video_id; no redundant inconsistent FK.
    analysis_job_id: Mapped[UUID] = mapped_column(ForeignKey("analysis_jobs.id", ondelete="RESTRICT"), primary_key=True)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    analysis_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    analysis_finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    fingerprint: Mapped[str] = mapped_column(String(64))
    roi_status: Mapped[str] = mapped_column(String(20))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    roi_snapshot: Mapped[dict] = mapped_column(JSONB)
    execution_snapshot: Mapped[dict] = mapped_column(JSONB)
    # Immutable managed CSV copies: relative paths, SHA-256, bytes, rows, null counts.
    artifacts: Mapped[dict] = mapped_column(JSONB)


class VisionZoneMetric(Base):
    __tablename__ = "vision_zone_metrics"
    __table_args__ = (
        CheckConstraint("video_offset_ms >= 0", name="ck_vision_zone_offset"),
        CheckConstraint("mean_person_count >= 0 AND inflow >= 0 AND outflow >= 0", name="ck_vision_zone_counts"),
        CheckConstraint("flow_angle_deg >= 0 AND flow_angle_deg < 360", name="ck_vision_zone_flow_angle"),
        CheckConstraint("magnitude_px >= 0 AND motion >= 0", name="ck_vision_zone_motion"),
        CheckConstraint("mean_angle_deg >= 0 AND mean_angle_deg < 360 AND direction_sample_count > 0", name="ck_vision_zone_direction"),
        CheckConstraint("(mean_person_count IS NULL AND inflow IS NULL AND outflow IS NULL) OR (mean_person_count IS NOT NULL AND inflow IS NOT NULL AND outflow IS NOT NULL)", name="ck_vision_zone_observation_group"),
        CheckConstraint("(flow_angle_deg IS NULL AND magnitude_px IS NULL AND motion IS NULL) OR (flow_angle_deg IS NOT NULL AND magnitude_px IS NOT NULL AND motion IS NOT NULL)", name="ck_vision_zone_flow_group"),
        CheckConstraint("(mean_angle_deg IS NULL) = (direction_sample_count IS NULL)", name="ck_vision_zone_direction_group"),
        Index("ix_vision_zone_imported", "imported_at"),
    )
    analysis_job_id: Mapped[UUID] = mapped_column(ForeignKey("vision_results.analysis_job_id", ondelete="RESTRICT"), primary_key=True)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, nullable=False)
    region_key: Mapped[str] = mapped_column(String(100), primary_key=True)
    video_offset_ms: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    # Full outer union preserves absent CSV buckets without invented zero values.
    has_observation: Mapped[bool] = mapped_column(Boolean)
    has_flow: Mapped[bool] = mapped_column(Boolean)
    has_individual_aggregate: Mapped[bool] = mapped_column(Boolean)
    mean_person_count: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    inflow: Mapped[int | None] = mapped_column(BigInteger)
    outflow: Mapped[int | None] = mapped_column(BigInteger)
    flow_angle_deg: Mapped[float | None] = mapped_column(Float)
    magnitude_px: Mapped[float | None] = mapped_column(Float)
    motion: Mapped[float | None] = mapped_column(Float)
    low_confidence: Mapped[bool | None] = mapped_column(Boolean)
    mean_angle_deg: Mapped[float | None] = mapped_column(Float)
    direction_sample_count: Mapped[int | None] = mapped_column(BigInteger, comment="CSV n_tracks: contributing rows, NOT unique people")


class VisionBottleneckScore(Base):
    __tablename__ = "vision_bottleneck_scores"
    __table_args__ = (
        CheckConstraint("window_start_ms >= 0", name="ck_vision_score_window"),
        CheckConstraint("score BETWEEN 0 AND 100 AND growth BETWEEN 0 AND 1 AND slowdown BETWEEN 0 AND 1 AND imbalance BETWEEN 0 AND 1", name="ck_vision_score_ranges"),
        CheckConstraint("(risk_level = '낮음' AND severity = 'low') OR (risk_level = '주의' AND severity = 'medium') OR (risk_level = '높음' AND severity = 'high')", name="ck_vision_score_risk_mapping"),
        Index("ix_vision_score_imported", "imported_at"),
    )
    analysis_job_id: Mapped[UUID] = mapped_column(ForeignKey("vision_results.analysis_job_id", ondelete="RESTRICT"), primary_key=True)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, nullable=False)
    region_key: Mapped[str] = mapped_column(String(100), primary_key=True)
    window_start_ms: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    score: Mapped[float] = mapped_column(Float)
    growth: Mapped[float] = mapped_column(Float)
    slowdown: Mapped[float] = mapped_column(Float)
    imbalance: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(10))
    severity: Mapped[str] = mapped_column(String(20))
