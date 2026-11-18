"""Append-only revisions, optimistic concurrency, legacy snapshot fallback."""
from copy import deepcopy
import hashlib,json
from uuid import uuid4
from sqlalchemy import select
from app.models import VideoRoiRevision
from .vision_csv import validate_roi
from .roi_geometry import validate_disjoint

def fingerprint(snapshot):
    return hashlib.sha256(json.dumps(snapshot,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def latest(db,video):
    row=db.scalar(select(VideoRoiRevision).where(VideoRoiRevision.video_id==video.id).order_by(VideoRoiRevision.ordinal.desc()).limit(1))
    return (row.snapshot,row.fingerprint,row.ordinal) if row else (video.roi_snapshot,fingerprint(video.roi_snapshot),0)
def append(db,video,snapshot,*,allow_approved=False):
    snap=deepcopy(snapshot)
    validate_roi(snap);validate_disjoint(snap['zones'])
    if not allow_approved and (snap['approved_for_analysis'] or snap['status']=='approved'):raise ValueError('User edits must be candidate revisions')
    if any(z['name']!=z['name'].strip() for z in snap['zones']):raise ValueError('Trim zone names')
    _,_,ordinal=latest(db,video)
    revision_id=uuid4();snap.update(schema_version=2,revision_id=str(revision_id),overlap_policy='reject',gap_policy='outside')
    row=VideoRoiRevision(id=revision_id,video_id=video.id,ordinal=ordinal+1,snapshot=snap,fingerprint=fingerprint(snap))
    db.add(row);db.flush();return row
