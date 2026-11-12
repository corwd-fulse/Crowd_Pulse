
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Video
import mimetypes

from fastapi.responses import FileResponse



router = APIRouter(
    prefix="/api/videos",
    tags=["Videos"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads" / "videos"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MiB


# =====================================
# Response Schema
# =====================================

class VideoResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    camera_id: int | None
    filename: str
    saved_filename: str
    file_size: int
    content_type: str | None
    duration_ms: int | None
    recorded_at: datetime | None
    uploaded_at: datetime
    status: str
    data_source: str


# =====================================
# 영상 업로드
# POST /api/videos/upload
# =====================================

@router.post("/upload", status_code=201)
def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    original_name = file.filename or ""
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="지원하지 않는 영상 형식입니다.",
        )

    video_id = uuid4().hex
    saved_name = f"{video_id}{extension}"
    saved_path = UPLOAD_DIR / saved_name

    total_size = 0

    try:
        with saved_path.open("wb") as destination:
            while chunk := file.file.read(1024 * 1024):
                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="영상 크기는 최대 200MB까지 허용됩니다.",
                    )

                destination.write(chunk)

        db.add(
            Video(
                id=UUID(hex=video_id),
                filename=original_name,
                saved_filename=saved_name,
                storage_path=f"uploads/videos/{saved_name}",
                file_size=total_size,
                content_type=file.content_type,
            )
        )

        db.commit()

    except SQLAlchemyError:
        db.rollback()
        saved_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=503,
            detail="영상 정보를 저장할 수 없습니다. 잠시 후 다시 시도하세요.",
        ) from None

    except Exception:
        db.rollback()
        saved_path.unlink(missing_ok=True)
        raise

    finally:
        file.file.close()

    return {
        "video_id": video_id,
        "filename": original_name,
        "saved_filename": saved_name,
        "file_size": total_size,
        "status": "uploaded",
        "data_source": "user_upload",
    }


@router.post("/{video_id}/analyze", deprecated=True)
def analyze_video(video_id: str):
    raise HTTPException(410, 'Use POST /api/analysis/jobs with video_id; synchronous inference has been retired')


# =====================================
# 업로드 영상 목록 조회
# GET /api/videos
# =====================================

@router.get(
    "",
    response_model=list[VideoResponse],
)
def get_videos(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    statement = (
        select(Video)
        .order_by(
            Video.uploaded_at.desc(),
            Video.id.desc(),
        )
        .offset(offset)
        .limit(limit)
    )

    return db.scalars(statement).all()


# =====================================
# 특정 영상 상세 조회
# GET /api/videos/{video_id}
# =====================================

@router.get(
    "/{video_id}",
    response_model=VideoResponse,
)
def get_video(
    video_id: UUID,
    db: Session = Depends(get_db),
):
    video = db.get(Video, video_id)

    if video is None:
        raise HTTPException(
            status_code=404,
            detail="영상을 찾을 수 없습니다.",
        )

    return video


# =====================================
# 영상 스트리밍
# GET /api/videos/{video_id}/stream
# =====================================

@router.get("/{video_id}/stream")
def stream_video(
    video_id: UUID,
    db: Session = Depends(get_db),
):
    # 1. DB에서 영상 정보 조회
    video = db.get(Video, video_id)

    if video is None:
        raise HTTPException(
            status_code=404,
            detail="영상 정보를 찾을 수 없습니다.",
        )

    if video.status != "uploaded":
        raise HTTPException(
            status_code=404,
            detail="사용할 수 없는 영상입니다.",
        )

    # 2. 허용된 영상 저장 디렉터리 확인
    upload_root = UPLOAD_DIR.resolve()
    video_path = (BASE_DIR / video.storage_path).resolve()

    # 저장 경로가 업로드 디렉터리 밖으로 벗어나는지 검증
    if not video_path.is_relative_to(upload_root):
        raise HTTPException(
            status_code=403,
            detail="허용되지 않은 영상 경로입니다.",
        )

    # 3. 실제 파일 존재 여부 확인
    if not video_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="영상 파일이 존재하지 않습니다.",
        )

    # 4. MIME 타입 결정
    media_type = (
        mimetypes.guess_type(video_path.name)[0]
        or "application/octet-stream"
    )

    # 5. 영상 응답 반환
    return FileResponse(
        path=video_path,
        media_type=media_type,
        filename=video.filename,
        content_disposition_type="inline",
    )
