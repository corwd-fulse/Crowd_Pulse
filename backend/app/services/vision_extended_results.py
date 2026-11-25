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
    if 'crowd_speed_individual' in manifest and pixels_per_meter is None:
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
            # FILE_ONLY_KINDS라 DB 칸은 비워 두고 CSV 파일(artifacts manifest의 path)로만 보관한다.
            crowd_speed_individual=None,
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


def replace_dmcount_estimates(db: Session, job_id: UUID, dmcount_csv: Path) -> VisionExtendedResult:
    """이미 저장된 확장 결과의 DM-Count 부분만 새 결과로 바꾼다(다른 지표는 그대로).

    DM-Count만 다시 계산한 경우용 — 예: YOLO 박스 밖 잔여(yolo_residual_count)를 나중에 추가로
    구한 경우. store_extended_result처럼 CSV를 검증·복사하고 히트맵을 새 import 디렉터리로
    옮긴다. 예전 import 디렉터리의 파일은 지우지 않는다(다른 지표의 manifest 경로가 거기를 가리킨다).
    """
    from app.services.vision_extended_csv import _read_csv_rows

    dmcount_csv = Path(dmcount_csv)
    with dmcount_csv.open('rb') as stream:
        digest = stream_sha256(stream)
    rows = _read_csv_rows(dmcount_csv, 'dmcount_estimates')
    extended = db.scalar(select(VisionExtendedResult).where(
        VisionExtendedResult.analysis_job_id == job_id).with_for_update())
    if extended is None:
        raise JobNotFound('No extended result for this analysis job; store the full bundle first')

    created_dir = None
    try:
        root = BASE_DIR / 'uploads/results' / str(job_id) / 'extended'
        root.mkdir(parents=True, exist_ok=True)
        created_dir = Path(tempfile.mkdtemp(prefix='import-', dir=root))
        destination = created_dir / 'dmcount_estimates.csv'
        shutil.copyfile(dmcount_csv, destination)
        with destination.open('rb') as stream:
            if stream_sha256(stream) != digest:
                raise InvalidExtendedResult('File changed during import')
        _persist_density_maps(rows, dmcount_csv.resolve().parent, created_dir)
        extended.artifacts = {**(extended.artifacts or {}), 'dmcount_estimates': {
            'sha256': digest, 'bytes': dmcount_csv.stat().st_size, 'rows': len(rows),
            'path': destination.relative_to(BASE_DIR).as_posix()}}
        extended.dmcount_estimates = rows
        db.commit()
        return extended
    except Exception:
        db.rollback()
        _cleanup_created_dir(created_dir)
        raise


def attach_file_only_artifact(db: Session, job_id: UUID, kind: str, csv_path: Path) -> VisionExtendedResult:
    """이미 저장된 확장 결과에 FILE_ONLY_KINDS 파일(예: crowd_speed_individual)을 나중에 붙인다.

    64MiB 상한 때문에 빼고 저장했던 결과를 위한 것. 파일을 검증·복사하고 manifest에 경로만
    추가한다(DB JSON 칸은 건드리지 않음). 이동속도는 보정값이 필요하므로 저장된 결과에
    pixels_per_meter가 있어야 한다.
    """
    from app.services.vision_extended_csv import FILE_ONLY_KINDS, count_valid_csv_rows

    if kind not in FILE_ONLY_KINDS:
        raise InvalidExtendedResult(f'{kind} is not a file-only kind')
    csv_path = Path(csv_path)
    with csv_path.open('rb') as stream:
        digest = stream_sha256(stream)
    rows = count_valid_csv_rows(csv_path, kind)
    extended = db.scalar(select(VisionExtendedResult).where(
        VisionExtendedResult.analysis_job_id == job_id).with_for_update())
    if extended is None:
        raise JobNotFound('No extended result for this analysis job; store the full bundle first')
    if kind == 'crowd_speed_individual' and extended.pixels_per_meter is None:
        raise InvalidExtendedResult('crowd_speed_individual file requires a stored pixels_per_meter')

    created_dir = None
    try:
        root = BASE_DIR / 'uploads/results' / str(job_id) / 'extended'
        root.mkdir(parents=True, exist_ok=True)
        created_dir = Path(tempfile.mkdtemp(prefix='import-', dir=root))
        destination = created_dir / f'{kind}.csv'
        shutil.copyfile(csv_path, destination)
        with destination.open('rb') as stream:
            if stream_sha256(stream) != digest:
                raise InvalidExtendedResult('File changed during import')
        extended.artifacts = {**(extended.artifacts or {}), kind: {
            'sha256': digest, 'bytes': csv_path.stat().st_size, 'rows': rows, 'file_only': True,
            'path': destination.relative_to(BASE_DIR).as_posix()}}
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
