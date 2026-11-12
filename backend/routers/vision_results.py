"""Explicit result import only; no inference scheduling or server path input."""
import json
from pathlib import Path
import tempfile
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AnalysisJob, VisionResult, VisionZoneMetric, VisionBottleneckScore
from app.services import vision_results as service
from app.services.vision_csv import ExecutionMetadata, InvalidResult, MAX_FILE_BYTES, SCHEMAS

router = APIRouter(prefix='/api/analysis/jobs', tags=['Vision results'])


@router.get('/{job_id}/detections')
def detections(job_id: UUID, db: Session = Depends(get_db)):
    from app.services import detection_view
    job = db.get(AnalysisJob, job_id)
    if job is None:
        raise HTTPException(404, 'Analysis job not found')
    try:
        return detection_view.for_job(job, db.get(VisionResult, job_id))
    except detection_view.DetectionUnavailable as exc:
        raise HTTPException(404, str(exc)) from None
    except (ValueError, KeyError, TypeError, OverflowError, OSError):
        raise HTTPException(422, 'Stored tracking CSV could not be loaded or validated') from None


@router.post('/{job_id}/results')
def import_results(
    job_id: UUID,
    response: Response,
    roi_snapshot: Annotated[str, Form()],
    execution_metadata: Annotated[str, Form()],
    crowd_observations: Annotated[UploadFile, File()],
    flow: Annotated[UploadFile, File()],
    individual_directions: Annotated[UploadFile, File()],
    individual_zone_agg: Annotated[UploadFile, File()],
    bottleneck_scores: Annotated[UploadFile, File()],
    db: Session = Depends(get_db),
):
    uploads = dict(zip(SCHEMAS, [crowd_observations, flow, individual_directions, individual_zone_agg, bottleneck_scores]))
    try:
        if len(roi_snapshot) > 1_000_000 or len(execution_metadata) > 1_000_000:
            raise InvalidResult('JSON metadata exceeds allowed size')
        roi = json.loads(roi_snapshot)
        metadata = ExecutionMetadata.model_validate_json(execution_metadata)
        with tempfile.TemporaryDirectory(prefix='vision-upload-') as temp:
            paths, total = {}, 0
            for kind, upload in uploads.items():
                path = Path(temp) / f'{kind}.csv'
                size = 0
                with path.open('wb') as destination:
                    while chunk := upload.file.read(1024*1024):
                        size += len(chunk)
                        total += len(chunk)
                        if size > MAX_FILE_BYTES or total > 256*1024*1024:
                            raise HTTPException(413, 'CSV upload exceeds size limit')
                        destination.write(chunk)
                paths[kind] = path
            result, created = service.store_result(db, job_id, paths, roi, metadata)
        response.status_code = 201 if created else 200
        return {'analysis_job_id': job_id, 'created': created, 'roi_status': result.roi_status, 'imported_at': result.imported_at}
    except service.JobNotFound as exc:
        raise HTTPException(404, str(exc)) from None
    except service.ResultConflict as exc:
        raise HTTPException(409, str(exc)) from None
    except (InvalidResult, ValidationError, json.JSONDecodeError, TypeError):
        raise HTTPException(422, 'Invalid CSV bundle, ROI snapshot, execution metadata, or source video') from None
    except (SQLAlchemyError, OSError):
        db.rollback()
        raise HTTPException(503, 'Result storage unavailable') from None
    finally:
        for upload in uploads.values():
            upload.file.close()


def get_result(job_id: UUID, db: Session):
    try:
        return service.result_for_job(db, job_id)
    except service.JobNotFound as exc:
        raise HTTPException(404, str(exc)) from None


def direction_presentation(result):
    snapshot = result.execution_snapshot
    quality = snapshot.get('direction_quality', 'unvalidated')
    review = snapshot.get('direction_review', {})
    confirmed = quality == 'validated' and result.roi_status == 'approved' and bool(review.get('reviewer')) and bool(review.get('evidence'))
    return {'quality': quality, 'confirmed_for_display': confirmed,
            'label': '검증된 방향 추정값' if confirmed else '미검증 또는 사용 제한 방향 추정값',
            'algorithm_version': snapshot.get('algorithm_version', 'legacy-unspecified'),
            'csv_schema_version': snapshot.get('csv_schema_version', '1')}


def component_status(result):
    profile = result.execution_snapshot.get('analysis_profile', 'full')
    return {key: ('not_run' if profile == 'fast' and key in ('optical_flow','bottleneck') else 'completed')
            for key in ('tracking','observations','individual_directions','optical_flow','bottleneck')}


@router.get('/{job_id}/results')
def result_summary(job_id: UUID, db: Session = Depends(get_db)):
    result = get_result(job_id, db)
    job = db.get(AnalysisJob, job_id)
    artifacts = {kind: {k:v for k,v in entry.items() if k != 'path'} | {
        'download_url': f'/api/analysis/jobs/{job_id}/results/artifacts/{kind}'
    } for kind,entry in result.artifacts.items()}
    return {
        'analysis_job_id': job_id, 'video_id': job.video_id, 'status': job.status, 'analysis_mode':job.analysis_mode,
        'analysis_profile': result.execution_snapshot.get('analysis_profile','full'), 'components': component_status(result),
        'roi_status': result.roi_status, 'roi_snapshot': result.roi_snapshot,
        'video_resolution': [result.width, result.height], 'execution_snapshot': result.execution_snapshot,
        'analysis_started_at': result.analysis_started_at, 'analysis_finished_at': result.analysis_finished_at,
        'imported_at': result.imported_at, 'artifacts': artifacts,
        'direction_presentation': direction_presentation(result),
        'units': {'video_offset_ms':'video-relative milliseconds', 'mean_person_count':'mean per-second occupancy',
                  'direction_sample_count':'contributing samples, not distinct people', 'motion':'normalized pixel displacement, not m/s',
                  'raw_speed_px':'displacement over observation window, not physical speed',
                  'score':'bottleneck research index, not congestion_percent'},
    }


@router.get('/{job_id}/results/zone-metrics')
def zone_metrics(job_id: UUID, region_key: str | None = None,
                 start_ms: int = Query(0, ge=0), end_ms: int | None = Query(None, ge=0),
                 limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    result = get_result(job_id, db)
    if end_ms is not None and end_ms < start_ms:
        raise HTTPException(422, 'end_ms must be >= start_ms')
    query = select(VisionZoneMetric).where(VisionZoneMetric.analysis_job_id == job_id, VisionZoneMetric.imported_at == result.imported_at, VisionZoneMetric.video_offset_ms >= start_ms)
    if end_ms is not None:
        query = query.where(VisionZoneMetric.video_offset_ms <= end_ms)
    if region_key is not None:
        query = query.where(VisionZoneMetric.region_key == region_key)
    rows = db.scalars(query.order_by(VisionZoneMetric.video_offset_ms, VisionZoneMetric.region_key).offset(offset).limit(limit)).all()
    presentation = direction_presentation(result)
    items = []
    for row in rows:
        item = {c.name:getattr(row,c.name) for c in row.__table__.columns}
        item['flow_direction_confirmed'] = presentation['confirmed_for_display'] and row.has_flow and row.flow_angle_deg is not None and row.low_confidence is False
        item['individual_direction_confirmed'] = presentation['confirmed_for_display'] and row.has_individual_aggregate and row.mean_angle_deg is not None
        items.append(item)
    return {'items': items, 'limit':limit, 'offset':offset, 'direction_presentation': presentation}


@router.get('/{job_id}/results/bottleneck-scores')
def bottleneck_scores(job_id: UUID, region_key: str | None = None,
                      limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    result = get_result(job_id, db)
    query = select(VisionBottleneckScore).where(VisionBottleneckScore.analysis_job_id == job_id, VisionBottleneckScore.imported_at == result.imported_at)
    if region_key is not None:
        query = query.where(VisionBottleneckScore.region_key == region_key)
    rows = db.scalars(query.order_by(VisionBottleneckScore.window_start_ms, VisionBottleneckScore.region_key).offset(offset).limit(limit)).all()
    return {'items': [{c.name:getattr(row,c.name) for c in row.__table__.columns} for row in rows], 'limit':limit, 'offset':offset,
            'component_status': component_status(result)['bottleneck']}


@router.get('/{job_id}/results/artifacts/{kind}')
def artifact(job_id: UUID, kind: str, db: Session = Depends(get_db)):
    result = get_result(job_id, db)
    if kind not in SCHEMAS or kind not in result.artifacts:
        raise HTTPException(404, 'Artifact not found')
    path = (service.BASE_DIR / result.artifacts[kind]['path']).resolve()
    if not path.is_relative_to((service.BASE_DIR / 'uploads/results').resolve()) or not path.is_file():
        raise HTTPException(404, 'Artifact file unavailable')
    return FileResponse(path, media_type='text/csv', filename=f'{kind}.csv')
