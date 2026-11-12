
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.analytics import AnalysisJob, Video


router = APIRouter(
    prefix="/api/analysis",
    tags=["Analysis"],
)


# =========================
# Request / Response
# =========================

class AnalysisJobCreate(BaseModel):
    video_id: UUID


class AnalysisJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    video_id: UUID
    status: str
    model_version: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None


# =========================
# 분석 작업 생성
# POST /api/analysis/jobs
# =========================

@router.post(
    "/jobs",
    response_model=AnalysisJobResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_analysis_job(
    payload: AnalysisJobCreate,
    db: Session = Depends(get_db),
):
    # 1. 분석할 영상 존재 여부 확인
    video = db.get(Video, payload.video_id)

    if video is None:
        raise HTTPException(
            status_code=404,
            detail="영상을 찾을 수 없습니다.",
        )

    if video.status != "uploaded":
        raise HTTPException(
            status_code=409,
            detail="분석할 수 없는 영상입니다.",
        )

    # 2. 이미 활성화된 분석 작업이 있는지 확인
    existing_job = db.scalar(
        select(AnalysisJob).where(
            AnalysisJob.video_id == video.id,
            AnalysisJob.status.in_(["queued", "running"]),
        )
    )

    if existing_job is not None:
        raise HTTPException(
            status_code=409,
            detail="이미 대기 중이거나 진행 중인 분석 작업이 있습니다.",
        )

    # 3. 새로운 분석 작업 생성
    # YOLO 모델 미연결이므로 queued 상태로 등록
    job = AnalysisJob(
        video_id=video.id,
        status="queued",
    )

    try:
        db.add(job)
        db.commit()
        db.refresh(job)

    except IntegrityError as exc:
        db.rollback()

        # 동시 요청으로 발생한 중복 작업 인덱스 충돌 처리
        constraint_name = getattr(
            getattr(exc.orig, "diag", None),
            "constraint_name",
            None,
        )

        if constraint_name == "uq_analysis_jobs_active_video":
            raise HTTPException(
                status_code=409,
                detail="이미 대기 중이거나 진행 중인 분석 작업이 있습니다.",
            ) from None

        raise

    except Exception:
        db.rollback()
        raise

    return job


# =========================
# 분석 작업 목록 조회
# GET /api/analysis/jobs
# =========================

@router.get(
    "/jobs",
    response_model=list[AnalysisJobResponse],
)
def get_analysis_jobs(
    video_id: UUID | None = None,
    db: Session = Depends(get_db),
):
    query = select(AnalysisJob)

    if video_id is not None:
        query = query.where(
            AnalysisJob.video_id == video_id
        )

    query = query.order_by(
        AnalysisJob.created_at.desc(),
        AnalysisJob.id.desc(),
    ).limit(100)

    return db.scalars(query).all()


# =========================
# 분석 작업 상세 조회
# GET /api/analysis/jobs/{job_id}
# =========================

@router.get(
    "/jobs/{job_id}",
    response_model=AnalysisJobResponse,
)
def get_analysis_job(
    job_id: UUID,
    db: Session = Depends(get_db),
):
    job = db.get(AnalysisJob, job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="분석 작업을 찾을 수 없습니다.",
        )

    return job
