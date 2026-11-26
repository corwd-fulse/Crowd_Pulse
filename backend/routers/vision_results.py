"""Explicit result import only; no inference scheduling or server path input."""
from collections import defaultdict
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
from app.services.vision_csv import ExecutionMetadata, InvalidResult, MAX_STREAM_FILE_BYTES as MAX_FILE_BYTES, SCHEMAS

router = APIRouter(prefix='/api/analysis/jobs', tags=['Vision results'])


@router.get('/{job_id}/replay')
def replay(job_id:UUID,start_ms:float|None=Query(None,ge=0,allow_inf_nan=False),end_ms:float|None=Query(None,ge=0,allow_inf_nan=False),db:Session=Depends(get_db)):
    from app.services import replay_view,detection_view
    job=db.get(AnalysisJob,job_id)
    if job is None:raise HTTPException(404,'Analysis job not found')
    try:
        if end_ms is not None and start_ms is None:start_ms=0
        if start_ms is not None and end_ms is not None and not start_ms<end_ms<=start_ms+30000:raise HTTPException(422,'Replay window must be greater than zero and at most 30 seconds')
        if start_ms is None:
            result=db.get(VisionResult,job_id)
            if detection_view.tracking_path(job,result).stat().st_size>8*1024**2 or result.execution_snapshot['frame_count']>500000:start_ms=0
        if start_ms is not None:
            from app.services import replay_index
            try:
                index=replay_index.ensure(job,db.get(VisionResult,job_id))
                return replay_index.window(index,start_ms,end_ms if end_ms is not None else start_ms+5000)
            except replay_index.IndexPending:
                return Response(content=json.dumps({'status':'indexing','retry_after_ms':500}),status_code=202,media_type='application/json',headers={'Retry-After':'1'})
            except replay_index.WindowTooDense as exc:raise HTTPException(413,str(exc)) from None
        return replay_view.for_job(job,db.get(VisionResult,job_id))
    except detection_view.DetectionUnavailable as exc:
        raise HTTPException(404,str(exc)) from None
    except (ValueError,KeyError,TypeError,OverflowError,OSError):
        raise HTTPException(422,'Stored replay tracking, frame index or ROI could not be validated') from None


@router.get('/{job_id}/detections')
def detections(job_id: UUID,start_ms:float|None=Query(None,ge=0,allow_inf_nan=False),end_ms:float|None=Query(None,ge=0,allow_inf_nan=False), db: Session = Depends(get_db)):
    from app.services import detection_view
    job = db.get(AnalysisJob, job_id)
    if job is None:
        raise HTTPException(404, 'Analysis job not found')
    try:
        if end_ms is not None and start_ms is None:start_ms=0
        if start_ms is not None and end_ms is not None and not start_ms<end_ms<=start_ms+30000:raise HTTPException(422,'Detection window must be greater than zero and at most 30 seconds')
        if start_ms is None:
            result=db.get(VisionResult,job_id)
            if detection_view.tracking_path(job,result).stat().st_size>8*1024**2 or result.execution_snapshot['frame_count']>500000:start_ms=0
        if start_ms is not None:
            from app.services import replay_index
            try:
                return replay_index.window(replay_index.ensure(job,db.get(VisionResult,job_id)),start_ms,end_ms if end_ms is not None else start_ms+5000,boxes_only=True)
            except replay_index.IndexPending:
                return Response(content=json.dumps({'status':'indexing','retry_after_ms':500}),status_code=202,media_type='application/json',headers={'Retry-After':'1'})
            except replay_index.WindowTooDense as exc:raise HTTPException(413,str(exc)) from None
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
                        if size > MAX_FILE_BYTES or total > 4*1024**3:
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
    statuses = {key: ('not_run' if profile == 'fast' and key in ('optical_flow','bottleneck') else 'completed')
                for key in ('tracking','observations','individual_directions','optical_flow','bottleneck')}
    coverage = result.execution_snapshot.get('settings', {}).get('hybrid', {}).get('bottleneck_status', [])
    if profile != 'fast' and coverage and any(r['status'] == 'unanalyzed' for r in coverage):
        statuses['bottleneck'] = 'partial' if any(r['status'] == 'analyzed' for r in coverage) else 'unanalyzed'
    return statuses


def bottleneck_diagnostics(result, db):
    """Explain existing results from saved metrics; never rewrite their snapshot."""
    from app.services.bottleneck_diagnostics import describe_bottleneck_windows
    query = select(VisionZoneMetric).where(VisionZoneMetric.analysis_job_id == result.analysis_job_id,
                                           VisionZoneMetric.imported_at == result.imported_at)
    hybrid = {(r['sec'], r['zone']): r for r in
              result.execution_snapshot.get('settings', {}).get('hybrid', {}).get('rows', [])}
    records = [dict(sec=row.video_offset_ms // 1000, zone=row.region_key, count=row.mean_person_count,
                    motion=row.motion, inflow=row.inflow, outflow=row.outflow, low_confidence=row.low_confidence,
                    hybrid=hybrid.get((row.video_offset_ms // 1000, row.region_key)))
               for row in db.scalars(query)]
    scores = db.scalars(select(VisionBottleneckScore).where(
        VisionBottleneckScore.analysis_job_id == result.analysis_job_id,
        VisionBottleneckScore.imported_at == result.imported_at)).all()
    return describe_bottleneck_windows(records, {(r.window_start_ms // 1000, r.region_key) for r in scores},
                                      analysis_profile=result.execution_snapshot.get('analysis_profile', 'full'))


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
        'bottleneck_status': bottleneck_diagnostics(result, db),
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
    hybrid_rows = result.execution_snapshot.get('settings', {}).get('hybrid', {}).get('rows', [])
    hybrid = {(r['sec'] * 1000, r['zone']): r for r in hybrid_rows}
    items = []
    for row in rows:
        item = {c.name:getattr(row,c.name) for c in row.__table__.columns}
        item['flow_direction_confirmed'] = presentation['confirmed_for_display'] and row.has_flow and row.flow_angle_deg is not None and row.low_confidence is False
        item['individual_direction_confirmed'] = presentation['confirmed_for_display'] and row.has_individual_aggregate and row.mean_angle_deg is not None
        item['hybrid'] = hybrid.get((row.video_offset_ms, row.region_key))
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


@router.get('/{job_id}/results/dmcount-estimates')
def dmcount_estimates(job_id: UUID, db: Session = Depends(get_db)):
    """DM-Count(density_estimation.py) 참고 추정치. YOLO+ByteTrack과 완전히 별개 모델이고
    정확도가 검증되지 않았으므로(실측 실험에서 영상 조건에 따라 10~120% 차이) 병목/위험도
    계산에 쓰이지 않는다 — accuracy_status로 항상 명시해 프론트가 참고용 배지를 띄우게 한다.
    같은 window_start/zone 기준으로 /results/zone-metrics의 YOLO 카운트와 나란히 비교하도록
    의도된 보조 엔드포인트."""
    from app.services import vision_extended_results as extended_service
    try:
        result = extended_service.extended_result_for_job(db, job_id)
    except extended_service.JobNotFound as exc:
        raise HTTPException(404, str(exc)) from None
    if result.dmcount_estimates is None:
        raise HTTPException(404, 'DM-Count was not run for this analysis job')
    return {
        'analysis_job_id': job_id,
        'items': result.dmcount_estimates,
        'accuracy_status': 'unverified',
        'note': ('Reference estimate only, independent of YOLO+ByteTrack — not validated against '
                 'ground truth and not used in bottleneck_scores or risk level calculations.'),
    }


@router.get('/{job_id}/results/dmcount-estimates/density-map/{filename}')
def dmcount_density_map(job_id: UUID, filename: str, db: Session = Depends(get_db)):
    """dmcount-estimates의 density_map_path가 가리키는 히트맵 PNG를 서빙한다. filename은
    그 엔드포인트가 돌려준 값에서 그대로 가져온 것이어야 하며(우리 쪽이 만든 window_<n>.png
    꼴), 다른 job의 파일이나 경로 조작을 걸러내기 위해 이 job 전용 디렉터리 밖은 404로 막는다
    (artifact() 엔드포인트와 같은 경로 검증 패턴)."""
    from app.services import vision_extended_results as extended_service
    try:
        result = extended_service.extended_result_for_job(db, job_id)
    except extended_service.JobNotFound as exc:
        raise HTTPException(404, str(exc)) from None
    if result.dmcount_estimates is None:
        raise HTTPException(404, 'DM-Count was not run for this analysis job')
    # 저장 시점의 실제 상대경로를 그대로 쓴다(import 임시 디렉터리 이름이 매번 랜덤이라
    # 고정 패턴으로 재구성할 수 없음, vision_extended_results._persist_density_maps 참고).
    # basename이 겹칠 일은 없다 — window_<n>.png는 job마다 별도 디렉터리에 저장된다.
    known_paths = {Path(row['density_map_path']).name: row['density_map_path']
                   for row in result.dmcount_estimates if row.get('density_map_path')}
    if filename not in known_paths:
        raise HTTPException(404, 'Density map image not found')
    path = (extended_service.BASE_DIR / known_paths[filename]).resolve()
    if not path.is_relative_to((extended_service.BASE_DIR / 'uploads/results').resolve()) or not path.is_file():
        raise HTTPException(404, 'Density map image unavailable')
    return FileResponse(path, media_type='image/png', filename=filename)


@router.get('/{job_id}/results/extended-artifacts/{kind}')
def extended_artifact(job_id: UUID, kind: str, db: Session = Depends(get_db)):
    """확장 분석 결과 파일 원본(CSV/JSON) 내려받기. 특히 crowd_speed_individual처럼 DB에 넣지
    않고 파일로만 보관하는 종류(vision_extended_csv.FILE_ONLY_KINDS)를 받는 유일한 경로다.
    artifact()와 같은 경로 검증(이 job의 results 디렉터리 밖은 404)."""
    from app.services import vision_extended_results as extended_service
    try:
        extended = extended_service.extended_result_for_job(db, job_id)
    except extended_service.JobNotFound as exc:
        raise HTTPException(404, str(exc)) from None
    entry = (extended.artifacts or {}).get(kind)
    if not isinstance(entry, dict) or not entry.get('path'):
        raise HTTPException(404, 'Artifact not found')
    path = (extended_service.BASE_DIR / entry['path']).resolve()
    if not path.is_relative_to((extended_service.BASE_DIR / 'uploads/results' / str(job_id)).resolve()) or not path.is_file():
        raise HTTPException(404, 'Artifact file unavailable')
    suffix = path.suffix.lower()
    return FileResponse(path, media_type='application/json' if suffix == '.json' else 'text/csv',
                        filename=f'{kind}{suffix}')


@router.get('/{job_id}/results/model-comparison')
def model_comparison(job_id: UUID, db: Session = Depends(get_db)):
    """같은 window_start(10초 단위)·zone 기준으로 YOLO 구역 카운트와 DM-Count 추정치를
    나란히 묶어서 낸다 — 프론트 'AI 모델 비교' 탭이 그래프/히트맵을 그리는 단일 소스.
    YOLO는 vision_zone_metrics(초 단위)를 그 윈도우 안에서 평균 내 맞춘다. 병목 점수에는
    어느 쪽도 반영하지 않으며, accuracy_status로 DM-Count 쪽이 미검증임을 항상 명시한다."""
    from app.services import vision_extended_results as extended_service
    from app.services.density_estimation import WINDOW_S

    result = get_result(job_id, db)
    try:
        extended = extended_service.extended_result_for_job(db, job_id)
    except extended_service.JobNotFound as exc:
        raise HTTPException(404, str(exc)) from None
    if extended.dmcount_estimates is None:
        raise HTTPException(404, 'DM-Count was not run for this analysis job')

    yolo_rows = db.scalars(select(VisionZoneMetric).where(
        VisionZoneMetric.analysis_job_id == job_id,
        VisionZoneMetric.imported_at == result.imported_at,
        VisionZoneMetric.has_observation.is_(True),
    )).all()
    yolo_by_window = defaultdict(list)
    for row in yolo_rows:
        window_start = (row.video_offset_ms // (WINDOW_S * 1000)) * WINDOW_S
        yolo_by_window[(window_start, row.region_key)].append(float(row.mean_person_count))

    items = []
    for rec in extended.dmcount_estimates:
        key = (rec['window_start'], rec['zone'])
        yolo_samples = yolo_by_window.get(key)
        density_map_path = rec.get('density_map_path')
        yolo_count = round(sum(yolo_samples) / len(yolo_samples), 2) if yolo_samples else None
        # YOLO 박스 밖에 남은 DM-Count density(예전 결과에는 없음 → None). YOLO+DM-Count 보완
        # 추정 = YOLO 인원 + 그 잔여: YOLO가 놓친 사람(먼 곳·겹침)을 DM-Count로 채운다는 가정.
        residual = rec.get('yolo_residual_count')
        items.append({
            'job_id': job_id,
            'zone': rec['zone'],
            'window_start': rec['window_start'],
            'yolo_count': yolo_count,
            'dmcount_residual_count': residual,
            'fused_count': round(yolo_count + residual, 2) if yolo_count is not None and residual is not None else None,
            'dmcount_estimated_count': rec['estimated_count'],
            'dmcount_inference_ms': rec['inference_ms'],
            'dmcount_model_version': rec['model_version'],
            'density_map_path': (f'/api/analysis/jobs/{job_id}/results/dmcount-estimates/density-map/'
                                  f'{Path(density_map_path).name}') if density_map_path else None,
        })
    return {
        'analysis_job_id': job_id,
        'items': items,
        'accuracy_status': 'unverified',
        'note': ('DM-Count is independent of YOLO+ByteTrack, not validated against ground truth, '
                 'and not used in bottleneck_scores or risk level calculations — comparison only.'),
    }


@router.get('/{job_id}/results/video-forecast')
def video_forecast(job_id: UUID, source: str = 'yolo', horizon: int = 60, db: Session = Depends(get_db)):
    """시계열 예측 탭: 10·30·60초 뒤 인원 예측(모델별 backtest)과 이상 흐름 3단계.
    forecast/run_job_forecast.py 를 그 job에 돌린 결과 파일을 읽는다. 돌리지 않았으면 404."""
    from app.services import video_forecast_results as forecast_service

    get_result(job_id, db)
    try:
        return forecast_service.video_forecast(job_id, source, horizon, db)
    except forecast_service.ForecastNotFound as exc:
        raise HTTPException(404, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


@router.get('/{job_id}/results/artifacts/{kind}')
def artifact(job_id: UUID, kind: str, db: Session = Depends(get_db)):
    result = get_result(job_id, db)
    if kind not in SCHEMAS or kind not in result.artifacts:
        raise HTTPException(404, 'Artifact not found')
    path = (service.BASE_DIR / result.artifacts[kind]['path']).resolve()
    if not path.is_relative_to((service.BASE_DIR / 'uploads/results').resolve()) or not path.is_file():
        raise HTTPException(404, 'Artifact file unavailable')
    return FileResponse(path, media_type='text/csv', filename=f'{kind}.csv')
