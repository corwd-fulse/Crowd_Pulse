"""Bounded raw-body uploads; avoids buffering long multipart files in memory."""
import math
import os
import shutil
from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from app.database import get_db
from app.models import Video
from routers.videos import UPLOAD_DIR, ALLOWED_EXTENSIONS

router = APIRouter(prefix='/api/capture', tags=['Capture uploads'])
MAX_BYTES = min(max(int(os.getenv('CAPTURE_MAX_UPLOAD_BYTES', str(2 * 1024**3))), 1), 8 * 1024**3)
RESERVE_BYTES = 1024**3


@router.get('/upload-policy')
def policy():
    return {'max_bytes': MAX_BYTES, 'extensions': sorted(ALLOWED_EXTENSIONS), 'reserve_bytes': RESERVE_BYTES,
        'transport': 'raw video body; one request per file; sequential client queue',
        'retry': 'retry whole failed upload; no resumable upload',
        'analysis': 'existing ROI approval and Worker queue; no automatic analysis',
        'tracking_replay_limit_bytes': 64 * 1024**2}


def probe(path):
    # Existing OpenCV dependency; metadata only, never decode all frames here.
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if cap.isOpened() and math.isfinite(fps) and math.isfinite(frames) and fps > 0 and frames > 0:
            return round(frames / fps * 1000)
        return None
    finally:
        cap.release()


@router.post('/videos/upload', status_code=201)
async def upload(request: Request, filename: str = Query(min_length=1, max_length=255), db: Session = Depends(get_db)):
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, 'Unsupported video extension')
    length = request.headers.get('content-length')
    if length:
        try:
            declared = int(length)
        except ValueError:
            raise HTTPException(400, 'Invalid content length') from None
        if declared < 1 or declared > MAX_BYTES:
            raise HTTPException(413, 'Empty or oversized video')
    if shutil.disk_usage(UPLOAD_DIR).free < RESERVE_BYTES + (int(length) if length else MAX_BYTES):
        raise HTTPException(507, 'Insufficient storage reserve')
    video_id = uuid4()
    saved_name = f'{video_id.hex}{extension}'
    path = UPLOAD_DIR / saved_name
    created = False
    total = 0
    committed = False
    try:
        with path.open('xb') as destination:
            created = True
            async for chunk in request.stream():
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(413, 'Video exceeds capture upload limit')
                if shutil.disk_usage(UPLOAD_DIR).free < RESERVE_BYTES + len(chunk):
                    raise HTTPException(507, 'Storage reserve exhausted')
                await run_in_threadpool(destination.write, chunk)
        if not total:
            raise HTTPException(400, 'Empty video')
        duration = await run_in_threadpool(probe, path)
        row = Video(id=video_id, filename=filename, saved_filename=saved_name,
            storage_path=f'uploads/videos/{saved_name}', file_size=total,
            content_type=request.headers.get('content-type'), duration_ms=duration)
        db.add(row)
        db.commit()
        committed = True
        return {'video_id': str(video_id), 'filename': filename, 'file_size': total,
            'duration_ms': duration, 'status': 'uploaded', 'analysis_status': 'not_requested'}
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(503, 'Video metadata storage unavailable') from None
    finally:
        if created and not committed:
            db.rollback()
            path.unlink(missing_ok=True)
