
from pathlib import Path
from uuid import uuid4
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.tracker import track_video

router = APIRouter(
    prefix="/api/videos",
    tags=["Videos"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads" / "videos"
RESULTS_DIR = BASE_DIR / "uploads" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB


@router.post("/upload", status_code=201)
def upload_video(file: UploadFile = File(...)):
    original_name = file.filename or ""

    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="지원하지 않는 영상 형식입니다.",
        )

    # 서버에 저장할 파일명은 UUID로 생성
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

    except Exception:
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


@router.post("/{video_id}/analyze")
def analyze_video(video_id: str):
    matches = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="업로드된 영상을 찾을 수 없습니다.")

    src_path = matches[0]
    summary = track_video(src_path, RESULTS_DIR, video_id)

    return {
        "video_id": video_id,
        "status": "analyzed",
        "tracker": "bytetrack",
        "model": "yolo11m.pt",
        **summary,
    }
