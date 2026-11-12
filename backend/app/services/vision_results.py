"""Owns one import transaction; never invokes AI or a worker."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from uuid import UUID

from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from app.models import AnalysisJob, Video, VisionResult, VisionZoneMetric, VisionBottleneckScore
from app.services.vision_csv import ExecutionMetadata, InvalidResult, RISK_MAPPING, validate_bundle

BASE_DIR = Path(__file__).resolve().parents[2]


class ResultConflict(ValueError):
    pass


class JobNotFound(LookupError):
    pass


def store_result(db: Session, job_id: UUID, paths: dict[str, Path], roi: dict, metadata: ExecutionMetadata, *, worker_token: UUID | None = None):
    roi_status, manifest, records = validate_bundle(paths, roi, metadata)
    try:
        # Keep historical FULL import fingerprints stable when the new optional
        # field was absent from the original metadata.
        execution_key = metadata.model_dump(mode='json')
        if metadata.analysis_profile == 'full':
            execution_key.pop('analysis_profile', None)
        canonical = json.dumps({'roi': roi, 'execution': execution_key, 'files': manifest}, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError):
        raise InvalidResult('Snapshots must contain finite JSON values') from None
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    created_dir = None
    try:
        # Serializes imports for a job, including simultaneous identical requests.
        job = db.scalar(select(AnalysisJob).where(AnalysisJob.id == job_id).with_for_update())
        if job is None:
            raise JobNotFound('Analysis job not found')
        existing = db.get(VisionResult, job_id)
        if existing:
            if existing.fingerprint != fingerprint:
                raise ResultConflict('A different result already exists for this job; create a new analysis job')
            db.commit()
            return existing, False
        if job.worker_token is not None and (worker_token != job.worker_token or job.status != 'running'):
            raise ResultConflict('Worker ownership lost or worker-owned job cannot be imported manually')
        if worker_token is not None and job.worker_token != worker_token:
            raise ResultConflict('Worker ownership lost')
        if metadata.analysis_mode != job.analysis_mode:
            raise ResultConflict('Execution mode differs from the analysis job')
        if metadata.analysis_profile != job.analysis_profile:
            raise ResultConflict('Execution profile differs from the analysis job')
        if job.status in ('failed', 'cancelled'):
            raise ResultConflict('Cannot import a result into a failed or cancelled job')
        if job.started_at is not None and job.started_at != metadata.analysis_started_at:
            raise ResultConflict('Actual analysis start differs from the existing job timestamp')
        if job.finished_at is not None and job.finished_at != metadata.analysis_finished_at:
            raise ResultConflict('Actual analysis finish differs from the existing job timestamp')
        video = db.get(Video, job.video_id)
        video_path = (BASE_DIR / video.storage_path).resolve()
        if not video_path.is_relative_to((BASE_DIR / 'uploads/videos').resolve()) or not video_path.is_file():
            raise InvalidResult('Registered source video is missing or outside uploads/videos')
        with video_path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != metadata.video_sha256:
                raise InvalidResult('Execution video hash differs from the registered video')
        root = BASE_DIR / 'uploads/results' / str(job_id)
        root.mkdir(parents=True, exist_ok=True)
        created_dir = Path(tempfile.mkdtemp(prefix='import-', dir=root))
        for kind, source in paths.items():
            destination = created_dir / f'{kind}.csv'
            shutil.copyfile(source, destination)
            # Hash copied bytes as well; reject changed source files (TOCTOU).
            with destination.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != manifest[kind]['sha256']:
                    raise InvalidResult('CSV changed during import')
            manifest[kind]['path'] = destination.relative_to(BASE_DIR).as_posix()
        result = VisionResult(
            analysis_job_id=job_id, analysis_started_at=metadata.analysis_started_at,
            analysis_finished_at=metadata.analysis_finished_at, fingerprint=fingerprint,
            roi_status=roi_status, width=roi['video_resolution'][0], height=roi['video_resolution'][1],
            roi_snapshot=roi, execution_snapshot=metadata.model_dump(mode='json'), artifacts=manifest,
        )
        db.add(result)
        db.flush()
        # Merge keys, not row positions. Missing aggregate buckets stay NULL.
        combined = {}
        for kind in ['crowd_observations', 'flow', 'individual_zone_agg']:
            for row in records[kind]:
                key = (row['sec'] * 1000, row['zone'])
                target = combined.setdefault(key, dict(
                    analysis_job_id=job_id, imported_at=result.imported_at,
                    video_offset_ms=key[0], region_key=key[1],
                    has_observation=False, has_flow=False, has_individual_aggregate=False,
                    mean_person_count=None, inflow=None, outflow=None, flow_angle_deg=None,
                    magnitude_px=None, motion=None, low_confidence=None,
                    mean_angle_deg=None, direction_sample_count=None,
                ))
                if kind == 'crowd_observations':
                    target.update(has_observation=True, mean_person_count=Decimal(str(row['count'])), inflow=row['inflow'], outflow=row['outflow'])
                elif kind == 'flow':
                    target.update(has_flow=True, flow_angle_deg=row['angle_deg'], magnitude_px=row['magnitude_px'], motion=row['motion'], low_confidence=row['low_confidence'])
                else:
                    target.update(has_individual_aggregate=True, mean_angle_deg=row['mean_angle_deg'], direction_sample_count=row['n_tracks'])
        zone_rows = list(combined.values())
        for offset in range(0, len(zone_rows), 1000):
            db.execute(insert(VisionZoneMetric), zone_rows[offset:offset+1000])
        score_rows = [dict(
            analysis_job_id=job_id, imported_at=result.imported_at, region_key=row['zone'],
            window_start_ms=row['window_start']*1000, score=row['score'], growth=row['growth'],
            slowdown=row['slowdown'], imbalance=row['imbalance'], risk_level=row['risk_level'],
            severity=RISK_MAPPING[row['risk_level']],
        ) for row in records['bottleneck_scores']]
        for offset in range(0, len(score_rows), 1000):
            db.execute(insert(VisionBottleneckScore), score_rows[offset:offset+1000])
        # Use caller-supplied actual execution timestamps, never import time.
        job.started_at = metadata.analysis_started_at
        job.finished_at = metadata.analysis_finished_at
        job.model_version = metadata.model_version
        job.status = 'succeeded'
        job.error_message = None
        db.commit()
        return result, True
    except Exception:
        db.rollback()
        if created_dir is not None:
            # Only the fresh directory allocated by this call is ever removed.
            resolved = created_dir.resolve()
            managed_root = (BASE_DIR / 'uploads/results').resolve()
            if resolved.is_relative_to(managed_root) and resolved != managed_root and created_dir.name.startswith('import-'):
                shutil.rmtree(created_dir)
        raise


def result_for_job(db: Session, job_id: UUID) -> VisionResult:
    if db.get(AnalysisJob, job_id) is None:
        raise JobNotFound('Analysis job not found')
    result = db.get(VisionResult, job_id)
    if result is None:
        raise JobNotFound('No imported result for this analysis job')
    return result
