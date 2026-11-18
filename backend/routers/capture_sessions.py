"""Venue-scoped capture registration and read-only training exports."""
import csv
import io
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select, func, cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import AnalysisJob, Video, VisionResult, VisionZoneMetric, VenueRevision, ConcentrationRun
from app.models.capture_session import CaptureSessionRevision, CaptureVideoVenue
from app.services.capture_sessions import CaptureData, COLUMNS, INTERVALS, training_rows, dataset_schema
from routers.venues import Configuration, SaveRevision, latest, save

router = APIRouter(prefix='/api/capture', tags=['Capture sessions'])


class VenueCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    address: str = Field(default='', max_length=500)
    venue_type: str = Field(default='public_space', min_length=1, max_length=80)


@router.get('/venues')
def venues(db: Session = Depends(get_db)):
    ranked = select(VenueRevision.venue_id, func.max(VenueRevision.revision).label('revision')).group_by(VenueRevision.venue_id).subquery()
    rows = db.scalars(select(VenueRevision).join(ranked, (VenueRevision.venue_id == ranked.c.venue_id) & (VenueRevision.revision == ranked.c.revision))).all()
    return [{'venue_id': r.venue_id, 'revision': r.revision, 'name': r.configuration['name'],
        'address': r.configuration.get('address', ''), 'venue_type': r.configuration.get('venue_type', 'unspecified')} for r in rows]


@router.post('/venues', status_code=201)
def create_venue(payload: VenueCreate, db: Session = Depends(get_db)):
    return save(str(uuid4()), SaveRevision(base_revision=0, configuration=Configuration(
        name=payload.name, address=payload.address, venue_type=payload.venue_type)), db)


@router.get('/venues/{venue_id}/jobs')
def venue_jobs(venue_id: str, db: Session = Depends(get_db)):
    from routers.analysis import AnalysisJobResponse
    venue = latest(db, venue_id)
    if venue is None:
        raise HTTPException(404, 'Venue not found')
    job_ids = {UUID(c['job_id']) for c in venue.configuration.get('cameras', []) if c.get('job_id')}
    job_ids |= {UUID(z['mapping']['job_id']) for z in venue.configuration.get('zones', []) if z.get('mapping')}
    videos = {UUID(c['video_id']) for r in sessions(venue_id, db) for c in r['data'].get('clips', []) if c.get('active', True)}
    if not job_ids and not videos:
        return []
    rows = db.scalars(select(AnalysisJob).where((AnalysisJob.id.in_(job_ids)) | (AnalysisJob.video_id.in_(videos)))
        .order_by(AnalysisJob.created_at.desc()).limit(200)).all()
    return [AnalysisJobResponse.model_validate(r) for r in rows]


@router.get('/venues/{venue_id}/sources')
def available_sources(venue_id: str, db: Session = Depends(get_db)):
    if latest(db, venue_id) is None:
        raise HTTPException(404, 'Venue not found')
    videos = db.scalars(select(Video).outerjoin(CaptureVideoVenue, Video.id == CaptureVideoVenue.video_id).where(
        (CaptureVideoVenue.venue_id == venue_id) | (CaptureVideoVenue.video_id.is_(None)))
        .order_by(Video.uploaded_at.desc()).limit(200)).all()
    video_ids = [v.id for v in videos]
    ranked = select(AnalysisJob.id, func.row_number().over(partition_by=AnalysisJob.video_id,
        order_by=AnalysisJob.created_at.desc()).label('rank')).where(AnalysisJob.video_id.in_(video_ids)).subquery()
    jobs = db.scalars(select(AnalysisJob).join(ranked, AnalysisJob.id == ranked.c.id)
        .where(ranked.c.rank <= 30).order_by(AnalysisJob.created_at.desc())).all()
    snapshots = dict(db.execute(select(VisionResult.analysis_job_id, VisionResult.roi_snapshot)
        .where(VisionResult.analysis_job_id.in_([j.id for j in jobs]))).all())
    jobs_by_video = {}
    for job in jobs:
        snapshot = snapshots.get(job.id) if job.status == 'succeeded' else None
        jobs_by_video.setdefault(job.video_id, []).append({'id': str(job.id), 'status': job.status,
            'analysis_mode': job.analysis_mode, 'roi_names': [z['name'] for z in snapshot['zones']] if snapshot else []})
    result = []
    for video in videos:
        result.append({'id': str(video.id), 'filename': video.filename, 'duration_ms': video.duration_ms,
            'jobs': jobs_by_video.get(video.id, [])})
    return result


class SessionSave(BaseModel):
    base_revision: int = Field(default=0, ge=0)
    venue_revision: int = Field(gt=0)
    data: CaptureData


def session_row(db, venue_id, session_id):
    row = db.scalar(select(CaptureSessionRevision).where(CaptureSessionRevision.session_id == session_id)
        .order_by(CaptureSessionRevision.revision.desc()).limit(1))
    if row is None or row.venue_id != venue_id:
        raise HTTPException(404, 'Capture session not found in this venue')
    return row


def view(row):
    data = CaptureData.model_validate(row.data)
    days = []
    day = data.start.date()
    while day <= data.end.date():
        days.append({'date': day.isoformat(), 'is_weekend': day.weekday() >= 5,
            'is_weekday': day.weekday() < 5, 'is_holiday': data.is_holiday if day == data.start.date() else None})
        day += timedelta(days=1)
    return {'session_id': row.session_id, 'revision': row.revision, 'venue_id': row.venue_id,
        'venue_revision': row.venue_revision, 'data': row.data, 'calendar_days': days, 'created_at': row.created_at}


@router.get('/venues/{venue_id}/sessions')
def sessions(venue_id: str, db: Session = Depends(get_db)):
    ranked = select(CaptureSessionRevision.session_id, func.max(CaptureSessionRevision.revision).label('revision')).where(
        CaptureSessionRevision.venue_id == venue_id).group_by(CaptureSessionRevision.session_id).subquery()
    rows = db.scalars(select(CaptureSessionRevision).join(ranked,
        (CaptureSessionRevision.session_id == ranked.c.session_id) & (CaptureSessionRevision.revision == ranked.c.revision))
        .order_by(CaptureSessionRevision.created_at.desc()).limit(200)).all()
    return [view(r) for r in rows]


@router.get('/venues/{venue_id}/sessions/{session_id}')
def get_session(venue_id: str, session_id: UUID, db: Session = Depends(get_db)):
    return view(session_row(db, venue_id, session_id))


def persist(db, venue_id, session_id, payload, current=None):
    if payload.base_revision != (current.revision if current else 0):
        raise HTTPException(409, 'Session changed; reload before saving')
    venue = db.get(VenueRevision, (venue_id, payload.venue_revision))
    if venue is None:
        raise HTTPException(404, 'Venue revision not found')
    zone_ids = {z['id'] for z in venue.configuration.get('zones', [])}
    fingerprints = set()
    ownership = []
    for clip in payload.data.clips:
        video = db.get(Video, clip.video_id)
        if video is None:
            raise HTTPException(422, 'Unknown video')
        owner = db.get(CaptureVideoVenue, clip.video_id)
        if owner and owner.venue_id != venue_id:
            raise HTTPException(409, 'Video already belongs to another venue')
        if owner is None:
            ownership.append(CaptureVideoVenue(video_id=clip.video_id, venue_id=venue_id, venue_revision=payload.venue_revision))
        if any(z is not None and z not in zone_ids for z in (clip.zone_id, clip.target_zone_id)):
            raise HTTPException(422, 'Zone does not belong to pinned venue revision')
        # A video already assigned to another venue cannot silently contaminate this venue.
        prior_assignment = db.scalar(select(CaptureSessionRevision.session_id).where(
            CaptureSessionRevision.venue_id != venue_id,
            cast(CaptureSessionRevision.data, JSONB).contains({'clips': [{'video_id': str(clip.video_id)}]})).limit(1)) if owner is None else None
        if prior_assignment is not None:
            raise HTTPException(409, 'Video already belongs to another venue')
        if clip.job_id:
            job, result = db.get(AnalysisJob, clip.job_id), db.get(VisionResult, clip.job_id)
            if not job or job.video_id != clip.video_id or job.status != 'succeeded' or not result:
                raise HTTPException(422, 'Analysis must be a completed result of this video')
            if clip.roi_name not in {z['name'] for z in result.roi_snapshot['zones']}:
                raise HTTPException(422, 'ROI is absent from immutable analysis snapshot')
            sha = result.execution_snapshot.get('video_sha256')
            if clip.active and sha and sha in fingerprints:
                raise HTTPException(422, 'Duplicate source content cannot represent independent cameras')
            if clip.active and sha:
                fingerprints.add(sha)
    row = CaptureSessionRevision(session_id=session_id, revision=payload.base_revision+1,
        venue_id=venue_id, venue_revision=payload.venue_revision, data=payload.data.model_dump(mode='json'))
    # All validation finishes before writes. A PK collision rolls back the whole save.
    db.add_all({owner.video_id: owner for owner in ownership}.values())
    db.add(row)
    try:
        db.commit()
        db.refresh(row)
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, 'Concurrent session save; reload') from None
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(503, 'Capture metadata storage unavailable') from None
    return view(row)


@router.post('/venues/{venue_id}/sessions', status_code=201)
def create_session(venue_id: str, payload: SessionSave, db: Session = Depends(get_db)):
    return persist(db, venue_id, uuid4(), payload)


@router.post('/venues/{venue_id}/sessions/{session_id}/revisions', status_code=201)
def revise_session(venue_id: str, session_id: UUID, payload: SessionSave, db: Session = Depends(get_db)):
    return persist(db, venue_id, session_id, payload, session_row(db, venue_id, session_id))


@router.get('/schema')
def schema():
    return {'session': CaptureData.model_json_schema(), 'dataset': dataset_schema()}


@router.get('/venues/{venue_id}/sessions/{session_id}/timeline')
def timeline(venue_id: str, session_id: UUID, db: Session = Depends(get_db)):
    row = session_row(db, venue_id, session_id)
    data = CaptureData.model_validate(row.data)
    cameras = []
    for clip in data.clips:
        if not clip.active:
            continue
        video = db.get(Video, clip.video_id)
        duration = video.duration_ms if video else None
        if duration is None and clip.job_id:
            result = db.get(VisionResult, clip.job_id)
            execution = result.execution_snapshot if result else {}
            fps, frames = execution.get('fps'), execution.get('frame_count')
            if fps and frames:
                duration = round(frames / fps * 1000)
        actual = data.mode == 'synchronized' and data.timing_basis == 'recorded' and clip.clock_verified
        start = (clip.capture_start + timedelta(milliseconds=clip.clock_offset_ms) - data.start).total_seconds() if actual else None
        cameras.append({'camera_id': clip.camera_id, 'video_id': str(clip.video_id),
            'start_seconds': start, 'end_seconds': start + duration/1000 if start is not None and duration else None,
            'duration_ms': duration, 'reason': 'clock_unverified_or_independent' if not actual else ('duration_unknown' if not duration else None)})
    return {'session_id': session_id, 'revision': row.revision, 'mode': data.mode,
        'origin': data.start.isoformat() if data.mode == 'synchronized' else None,
        'duration_seconds': (data.end-data.start).total_seconds(), 'cameras': cameras,
        'missing_policy': 'Outside each recorded interval: missing observation; never zero-fill or extrapolate'}


class SessionEstimate(BaseModel):
    session_revision: int = Field(gt=0)
    offset_seconds: float = Field(ge=0, allow_inf_nan=False)
    target_zone_id: str | None = None


@router.post('/venues/{venue_id}/sessions/{session_id}/concentration', status_code=201)
def estimate_session(venue_id: str, session_id: UUID, payload: SessionEstimate, db: Session = Depends(get_db)):
    from routers.concentration import configuration, contexts_for, select_target
    from app.services.concentration import estimator
    row = session_row(db, venue_id, session_id)
    if payload.session_revision != row.revision:
        raise HTTPException(409, 'Session changed; reload before estimating')
    data = CaptureData.model_validate(row.data)
    if data.mode != 'synchronized':
        raise HTTPException(422, 'Session-based actual aggregation requires verified synchronized recordings')
    if payload.offset_seconds >= (data.end-data.start).total_seconds():
        raise HTTPException(422, 'Observation time is outside the capture session')
    config = select_target(configuration(db, venue_id, row.venue_revision), payload.target_zone_id)
    clips = {c.camera_id: c for c in data.clips if c.active}
    for camera in config['cameras']:
        clip = clips.get(camera['camera_id'])
        if not clip or str(clip.job_id) != camera['job_id'] or clip.roi_name != camera['roi_name']:
            raise HTTPException(422, 'Each target camera requires the same calibrated analysis/ROI in the session; configure a new map revision before binding another video')
        camera['capture_start'] = (clip.capture_start + timedelta(milliseconds=clip.clock_offset_ms)).isoformat()
        camera['clock_verified'] = clip.clock_verified
        camera['clock_evidence'] = clip.clock_evidence
    config['integration_mode'] = 'synchronized'
    request = {'revision': row.venue_revision, 'offset_seconds': payload.offset_seconds,
        'at': (data.start+timedelta(seconds=payload.offset_seconds)).isoformat(), 'target_zone_id': payload.target_zone_id}
    result = estimator.estimate(config, contexts_for(db, config), request)
    result.update(venue_id=venue_id, revision=row.venue_revision, session_id=str(session_id), session_revision=row.revision,
        venue_name=config['name'], registered_address=config['address'] or None, center_zone_id=config['center_zone_id'])
    run = ConcentrationRun(venue_id=venue_id, revision=row.venue_revision,
        request={**request, 'session_id': str(session_id), 'session_revision': row.revision}, result=result)
    db.add(run)
    try:
        db.commit()
        db.refresh(run)
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(503, 'Concentration evidence storage unavailable') from None
    return {'run_id': run.id, 'created_at': run.created_at, **result}


@router.get('/venues/{venue_id}/sessions/{session_id}/dataset.csv')
def export_dataset(venue_id: str, session_id: UUID, interval: int = 5,
        camera_id: str | None = None, zone_id: str | None = None, db: Session = Depends(get_db)):
    if interval not in INTERVALS:
        raise HTTPException(422, 'Interval must be 1, 5, 10, 30 or 60 seconds')
    row = session_row(db, venue_id, session_id)
    data = CaptureData.model_validate(row.data)
    import tempfile
    from pathlib import Path
    from fastapi.responses import FileResponse
    from starlette.background import BackgroundTask
    output = tempfile.NamedTemporaryFile(mode="w",encoding="utf-8-sig",newline="",suffix=".csv",prefix="capture-export-",delete=False)
    try:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        count = 0
        for clip in data.clips:
            if not clip.active or not clip.job_id or (camera_id and clip.camera_id != camera_id) or (zone_id and clip.zone_id != zone_id):
                continue
            result = db.get(VisionResult, clip.job_id)
            if result is None:
                raise HTTPException(409, 'Bound result is unavailable')
            metrics = db.scalars(select(VisionZoneMetric).where(VisionZoneMetric.analysis_job_id == clip.job_id,
                VisionZoneMetric.imported_at == result.imported_at, VisionZoneMetric.region_key == clip.roi_name)
                .order_by(VisionZoneMetric.video_offset_ms).limit(250001).execution_options(yield_per=1000))
            identity = {'session_id': str(row.session_id), 'session_revision': row.revision, 'venue_id': venue_id,
                'venue_revision': row.venue_revision, 'result_fingerprint': result.fingerprint}
            execution = result.execution_snapshot
            duration_ms = execution.get('frame_count', 0) / execution['fps'] * 1000 if execution.get('fps') else None
            for item in training_rows(({'video_offset_ms': m.video_offset_ms,
                    'mean_person_count': m.mean_person_count, 'inflow': m.inflow, 'outflow': m.outflow,
                    'motion': m.motion, 'low_confidence': m.low_confidence,
                    'complete_second': duration_ms is not None and m.video_offset_ms+1000 <= duration_ms}
                    for m in metrics), clip, data, identity, interval,ordered=True):
                if item['timestamp']:
                    end = datetime.fromisoformat(item['timestamp'])
                    if end > data.end or end - timedelta(seconds=interval) < data.start:
                        continue
                count += 1
                if count > 250000:
                    raise HTTPException(413, 'Export too large; filter by camera or zone')
                writer.writerow(item)
                if output.tell() > 256 * 1024**2:
                    raise HTTPException(413, 'Export text exceeds memory budget; filter by camera or zone')
    except ValueError:
        output.close();Path(output.name).unlink(missing_ok=True)
        raise HTTPException(413,'Export validation or row budget exceeded; split the session or filter by camera') from None
    except Exception:
        output.close();Path(output.name).unlink(missing_ok=True);raise
    output.close()
    return FileResponse(output.name,media_type='text/csv',filename=f'capture-{session_id}-{interval}s.csv',
        background=BackgroundTask(Path(output.name).unlink,missing_ok=True),
        headers={'X-Capture-Rows':str(count),'X-Forecast-Status':'features-only-no-model-predictions'})

