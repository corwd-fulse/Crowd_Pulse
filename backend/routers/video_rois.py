"""Explicit ROI registration; never reads or approves a JSON file automatically."""
from uuid import UUID
import json

from fastapi import APIRouter, Body, Depends, HTTPException, Header
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.database import get_db
from app.models import Video
from app.services.vision_csv import InvalidResult, validate_roi

router = APIRouter(prefix='/api/videos', tags=['Video ROI'])


@router.put('/{video_id}/roi')
def set_roi(video_id: UUID, snapshot: dict = Body(...), db: Session = Depends(get_db),
            if_none_match: str | None = Header(default=None)):
    # Serialize conditional creation. Existing clients retain their PUT contract.
    video = db.scalar(select(Video).where(Video.id == video_id).with_for_update())
    if video is None:
        raise HTTPException(404, 'Video not found')
    if if_none_match is not None and if_none_match != '*':
        raise HTTPException(400, 'Only If-None-Match: * is supported')
    if if_none_match == '*' and video.roi_snapshot is not None:
        raise HTTPException(412, 'ROI already exists; existing snapshot was preserved')
    try:
        state, _ = validate_roi(snapshot)
        json.dumps(snapshot, allow_nan=False)
    except (InvalidResult, ValueError, TypeError):
        raise HTTPException(422, 'Invalid ROI snapshot or approval status') from None
    video.roi_snapshot = snapshot
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(503, 'ROI storage unavailable') from None
    return {'video_id': video.id, 'roi_status': state, 'roi_snapshot': video.roi_snapshot}


@router.get('/{video_id}/roi')
def get_roi(video_id: UUID, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if video is None or video.roi_snapshot is None:
        raise HTTPException(404, 'Video or registered ROI not found')
    return {'video_id':video.id, 'roi_snapshot':video.roi_snapshot}
