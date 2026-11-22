"""Supplementary Vision AI metrics: density heatmap, dwell time, counter flow, queue
analysis, tracking quality, calibrated crowd speed.

These are additive extensions beyond the original FR-ZON spec (see pipeline.py
run_extended_zone_pipeline()) and are never required for an analysis job to succeed, so this
gets one JSONB-based table instead of five fully normalized ones with per-row check
constraints like vision.py's VisionZoneMetric/VisionBottleneckScore. Only written when a
VisionResult already exists for the same analysis_job_id (enforced by the FK below).
"""
from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class VisionExtendedResult(Base):
    __tablename__ = "vision_extended_results"
    __table_args__ = (
        CheckConstraint("pixels_per_meter IS NULL OR pixels_per_meter > 0", name="ck_vision_extended_calibration"),
        CheckConstraint(
            "(crowd_speed_zone IS NULL) = (pixels_per_meter IS NULL)",
            name="ck_vision_extended_speed_requires_calibration",
        ),
    )
    analysis_job_id: Mapped[UUID] = mapped_column(ForeignKey("vision_results.analysis_job_id", ondelete="RESTRICT"), primary_key=True)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # 보정값 없으면 속도 관련 열은 전부 NULL(위 체크 제약) — 보정 없이 환산한 값을 사실처럼 저장하지 않는다.
    pixels_per_meter: Mapped[float | None] = mapped_column(Float)
    # VisionResult.artifacts와 같은 모양(파일별 sha256/bytes/rows/path).
    artifacts: Mapped[dict] = mapped_column(JSONB)
    dwell_times: Mapped[list] = mapped_column(JSONB)
    counter_flow: Mapped[list] = mapped_column(JSONB)
    queue_scores: Mapped[list] = mapped_column(JSONB)
    density_heatmap: Mapped[list] = mapped_column(JSONB)
    tracking_quality: Mapped[dict] = mapped_column(JSONB)
    # none_as_null=True: SQLAlchemy의 JSON(B) 타입은 기본값(False)이면 Python None을 SQL NULL이
    # 아니라 JSON 리터럴 null로 저장한다 — 그러면 "crowd_speed_zone IS NULL" 체크 제약이
    # (의도와 달리) 항상 거짓이 되어 위 ck_vision_extended_speed_requires_calibration이
    # pixels_per_meter=None인 정상 케이스에서도 위반된다(실제 DB로 테스트하다 발견한 버그).
    crowd_speed_zone: Mapped[list | None] = mapped_column(JSONB(none_as_null=True))
    crowd_speed_individual: Mapped[list | None] = mapped_column(JSONB(none_as_null=True))
    # DM-Count(density_estimation.py)는 YOLO+ByteTrack과 완전히 독립된 모델이라 다른 보조
    # 지표처럼 '계산됐으면 항상 있음'이 아니라 '대부분의 job에서는 아예 안 돈다'(느린 CPU
    # 추론이라 opt-in). crowd_speed_zone과 같은 이유로 nullable — 정확도 미검증 참고
    # 지표이므로 bottleneck_scores 등 어떤 계산에도 입력으로 쓰지 않는다.
    dmcount_estimates: Mapped[list | None] = mapped_column(JSONB(none_as_null=True))
