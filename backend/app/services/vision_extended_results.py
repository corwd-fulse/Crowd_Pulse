"""Stores the supplementary Vision AI metrics from pipeline.run_extended_zone_pipeline().

Separate from vision_results.py on purpose: that module owns the core five-file FR-ZON
contract and the analysis_jobs status transition (job.status = 'succeeded'). This module
never touches job status — storing (or failing to store) extended metrics must not change
whether the core analysis job succeeded or failed (see worker.py, where this runs after
vision_results.store_result() has already committed).
"""
from pathlib import Path
import hashlib
import shutil
import tempfile
from uuid import UUID

from sqlalchemy.orm import Session

from app.models import AnalysisJob, VisionExtendedResult, VisionResult
from app.services.vision_extended_csv import InvalidExtendedResult, validate_extended_bundle
from app.services.vision_results import BASE_DIR, JobNotFound, ResultConflict


def store_extended_result(
    db: Session,
    job_id: UUID,
    paths: dict[str, Path],
    *,
    pixels_per_meter: float | None = None,
):
    manifest, records = validate_extended_bundle(paths)
    has_speed = 'crowd_speed_zone' in records
    if has_speed != (pixels_per_meter is not None):
        raise InvalidExtendedResult('crowd_speed_zone file requires pixels_per_meter and vice versa')
    # crowd_speed.py/pipeline.py allow a per-zone dict calibration, but this table's
    # pixels_per_meter column is a single Float — a dict here would only fail later with a
    # confusing driver-level type error, so reject it clearly up front.
    if pixels_per_meter is not None and not isinstance(pixels_per_meter, (int, float)):
        raise InvalidExtendedResult('Only a single pixels_per_meter value is stored; per-zone calibration is CSV-only for now')

    result = db.get(VisionResult, job_id)
    if result is None:
        raise JobNotFound('No core vision result for this analysis job; store that first')
    if db.get(VisionExtendedResult, job_id) is not None:
        raise ResultConflict('Extended result already exists for this analysis job')

    created_dir = None
    try:
        root = BASE_DIR / 'uploads/results' / str(job_id) / 'extended'
        root.mkdir(parents=True, exist_ok=True)
        created_dir = Path(tempfile.mkdtemp(prefix='import-', dir=root))
        for kind, source in paths.items():
            suffix = '.json' if kind == 'tracking_quality' else '.csv'
            destination = created_dir / f'{kind}{suffix}'
            shutil.copyfile(source, destination)
            with destination.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != manifest[kind]['sha256']:
                    raise InvalidExtendedResult('File changed during import')
            manifest[kind]['path'] = destination.relative_to(BASE_DIR).as_posix()

        extended = VisionExtendedResult(
            analysis_job_id=job_id,
            pixels_per_meter=pixels_per_meter,
            artifacts=manifest,
            dwell_times=records['dwell_times'],
            counter_flow=records['counter_flow'],
            queue_scores=records['queue_scores'],
            density_heatmap=records['density_heatmap'],
            tracking_quality=records['tracking_quality'],
            crowd_speed_zone=records.get('crowd_speed_zone'),
            crowd_speed_individual=records.get('crowd_speed_individual'),
        )
        db.add(extended)
        db.commit()
        return extended
    except Exception:
        db.rollback()
        if created_dir is not None:
            resolved = created_dir.resolve()
            managed_root = (BASE_DIR / 'uploads/results').resolve()
            if resolved.is_relative_to(managed_root) and resolved != managed_root and created_dir.name.startswith('import-'):
                shutil.rmtree(created_dir)
        raise


def extended_result_for_job(db: Session, job_id: UUID) -> VisionExtendedResult:
    if db.get(AnalysisJob, job_id) is None:
        raise JobNotFound('Analysis job not found')
    result = db.get(VisionExtendedResult, job_id)
    if result is None:
        raise JobNotFound('No imported extended result for this analysis job')
    return result
