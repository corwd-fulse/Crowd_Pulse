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

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AnalysisJob, VisionExtendedResult, VisionResult
from app.services.vision_extended_csv import InvalidExtendedResult, validate_extended_bundle
from app.services.vision_csv import stream_sha256
from app.services.vision_results import BASE_DIR, JobNotFound, ResultConflict, _cleanup_created_dir


def _persist_density_maps(records: list[dict] | None, trusted_root: Path, created_dir: Path) -> None:
    """dmcount_estimates 행의 density_map_path(density_estimation.py가 pipeline의 임시
    out 디렉터리 밑에 써둔 PNG)를 이 job의 영구 저장 디렉터리로 복사하고, 각 행의 경로를
    BASE_DIR 기준 상대경로로 바꿔치기한다 — artifacts manifest의 CSV path 처리와 같은 패턴.
    trusted_root 밖을 가리키면(worker.py가 넘긴 CSV와 다른 디렉터리 트리) 거부한다 — 이
    값은 density_estimation.py가 로컬에서 만든 경로지 외부 입력이 아니지만, 혹시라도
    조작되었을 때 임의 경로 복사를 막기 위한 방어다. 같은 윈도우의 zone들은 같은 이미지를
    가리키므로 한 번만 복사한다."""
    if not records:
        return
    copied: dict[str, str] = {}
    for row in records:
        src = row.get('density_map_path')
        if not src:
            continue
        src_path = Path(src).resolve()
        if not src_path.is_relative_to(trusted_root) or not src_path.is_file():
            raise InvalidExtendedResult('dmcount_estimates: density_map_path outside expected directory')
        if src not in copied:
            dest_root = created_dir / 'dmcount_heatmaps'
            dest_root.mkdir(parents=True, exist_ok=True)
            destination = dest_root / src_path.name
            shutil.copyfile(src_path, destination)
            copied[src] = destination.relative_to(BASE_DIR).as_posix()
        row['density_map_path'] = copied[src]


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
    # crowd_speed_individual is only ever produced alongside crowd_speed_zone (pipeline.py
    # only writes it when pixels_per_meter is a single float), but nothing enforces that here —
    # without this check a caller could persist speed_mps values with no calibration on record.
    if 'crowd_speed_individual' in records and pixels_per_meter is None:
        raise InvalidExtendedResult('crowd_speed_individual file requires pixels_per_meter')
    # crowd_speed.py/pipeline.py allow a per-zone dict calibration, but this table's
    # pixels_per_meter column is a single Float — a dict here would only fail later with a
    # confusing driver-level type error, so reject it clearly up front.
    if pixels_per_meter is not None and not isinstance(pixels_per_meter, (int, float)):
        raise InvalidExtendedResult('Only a single pixels_per_meter value is stored; per-zone calibration is CSV-only for now')

    # Locks the parent row (same pattern as vision_results.store_result locking AnalysisJob)
    # so two concurrent/retried imports for the same job serialize here instead of both
    # passing the "does it already exist" check below and racing on the insert.
    result = db.scalar(select(VisionResult).where(VisionResult.analysis_job_id == job_id).with_for_update())
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
                if stream_sha256(stream) != manifest[kind]['sha256']:
                    raise InvalidExtendedResult('File changed during import')
            manifest[kind]['path'] = destination.relative_to(BASE_DIR).as_posix()

        if 'dmcount_estimates' in paths:
            _persist_density_maps(records.get('dmcount_estimates'), paths['dmcount_estimates'].resolve().parent, created_dir)

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
            # crowd_speed_zone과 달리 pixels_per_meter 보정과 무관하게 독립적으로 켜고 끄는
            # 기능이라(YOLO 파이프라인과 완전히 별개 모델) 별도 required-pairing 검증 없음.
            dmcount_estimates=records.get('dmcount_estimates'),
        )
        db.add(extended)
        db.commit()
        return extended
    except Exception:
        db.rollback()
        _cleanup_created_dir(created_dir)
        raise


def extended_result_for_job(db: Session, job_id: UUID) -> VisionExtendedResult:
    if db.get(AnalysisJob, job_id) is None:
        raise JobNotFound('Analysis job not found')
    result = db.get(VisionExtendedResult, job_id)
    if result is None:
        raise JobNotFound('No imported extended result for this analysis job')
    return result
