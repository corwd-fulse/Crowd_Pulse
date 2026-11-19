"""Explicit ROI registration; never reads or approves a JSON file automatically."""
from uuid import UUID
import json
from copy import deepcopy

from fastapi import APIRouter, Body, Depends, HTTPException, Header
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.database import get_db
from app.models import Video
from app.services.vision_csv import InvalidResult, validate_roi
from app.services import roi_revisions

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
    existing,_,_=roi_revisions.latest(db,video)
    if if_none_match == '*' and existing is not None:
        raise HTTPException(412, 'ROI already exists; existing snapshot was preserved')
    if existing is not None:
        # Retain the explicit legacy approval operation without overwriting its
        # candidate. Geometry/display edits must use the conditional revision API.
        if (snapshot.get('status')=='approved' and snapshot.get('approved_for_analysis') is True
                and snapshot.get('zones')==existing.get('zones')
                and snapshot.get('video_resolution')==existing.get('video_resolution')):
            if existing['approved_for_analysis']:
                return {'video_id':video.id,'roi_status':'approved','roi_snapshot':existing,
                        'fingerprint':roi_revisions.fingerprint(existing)}
            approved=deepcopy(existing);approved.update(status='approved',approved_for_analysis=True)
            try:
                row=roi_revisions.append(db,video,approved,allow_approved=True);db.commit()
            except (InvalidResult,ValueError,TypeError):
                raise HTTPException(422,'Invalid approval geometry') from None
            except SQLAlchemyError:
                db.rollback();raise HTTPException(503,'ROI storage unavailable') from None
            return {'video_id':video.id,'roi_status':'approved','roi_snapshot':row.snapshot,
                    'fingerprint':row.fingerprint,'revision':row.ordinal}
        raise HTTPException(409, 'Existing ROI is immutable; append a revision instead')
    try:
        state, _ = validate_roi(snapshot)
        json.dumps(snapshot, allow_nan=False)
        if snapshot.get('schema_version')==2:
            from app.services.roi_geometry import validate_disjoint
            validate_disjoint(snapshot['zones'])
            if snapshot['video_resolution']!=video_resolution(video):raise ValueError('Resolution mismatch')
    except (InvalidResult, ValueError, TypeError):
        raise HTTPException(422, 'Invalid ROI snapshot or approval status') from None
    video.roi_snapshot = snapshot
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(503, 'ROI storage unavailable') from None
    return {'video_id': video.id, 'roi_status': state, 'roi_snapshot': video.roi_snapshot,
            'fingerprint':roi_revisions.fingerprint(video.roi_snapshot),'revision':0}


@router.get('/{video_id}/roi')
def get_roi(video_id: UUID, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(404, 'Video or registered ROI not found')
    snapshot,etag,ordinal=roi_revisions.latest(db,video)
    if snapshot is None:raise HTTPException(404, 'Video or registered ROI not found')
    return {'video_id':video.id, 'roi_snapshot':snapshot,'fingerprint':etag,'revision':ordinal}


def video_resolution(video):
    from app.services.vision_results import BASE_DIR
    import cv2
    path=(BASE_DIR/video.storage_path).resolve()
    if not path.is_relative_to((BASE_DIR/'uploads/videos').resolve()):raise ValueError('Invalid video path')
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():raise ValueError('Cannot inspect video')
        return [int(cap.get(3)),int(cap.get(4))]
    finally:cap.release()


@router.post('/{video_id}/roi-revisions',status_code=201)
def create_revision(video_id:UUID,snapshot:dict=Body(...),if_match:str|None=Header(default=None),db:Session=Depends(get_db)):
    video=db.scalar(select(Video).where(Video.id==video_id).with_for_update())
    if video is None:raise HTTPException(404,'Video not found')
    if if_match is None:raise HTTPException(428,'If-Match fingerprint required')
    _,etag,_=roi_revisions.latest(db,video)
    if if_match!=etag:raise HTTPException(412,'ROI changed; reload before saving a new revision')
    try:
        if snapshot.get('video_resolution')!=video_resolution(video):raise ValueError('Video resolution mismatch')
        row=roi_revisions.append(db,video,snapshot)
        db.commit()
    except (InvalidResult,ValueError,TypeError,KeyError):
        raise HTTPException(422,'Invalid candidate ROI, geometry, display settings or video resolution') from None
    except SQLAlchemyError:
        db.rollback();raise HTTPException(503,'ROI revision storage unavailable') from None
    return {'video_id':video.id,'roi_snapshot':row.snapshot,'fingerprint':row.fingerprint,'revision':row.ordinal}
