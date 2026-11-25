"""끊긴 확장 분석(extended pipeline)을 이어서 끝내고 DB에 저장하는 CLI.

확장 분석은 core 결과가 저장되어 job이 succeeded가 된 *뒤에* 돌기 때문에, 여기서 worker가
죽으면 worker의 recover_interrupted()는 이 job을 다시 잡지 않는다. 이 명령이 그 빈자리를
채운다 — worker가 남긴 작업 디렉터리(uploads/analysis/<job>/<attempt>/)를 찾아
run_extended_zone_pipeline(resumable=True)을 다시 호출하면, 끝난 단계는 건너뛰고 DM-Count는
마지막으로 저장된 10초 윈도우 다음부터 이어서 계산한다.

다른 컴퓨터(다른 GPU)로 옮겨 이어가기: DM-Count를 뺀 나머지 지표가 이미 끝난 상태라면
(extended/extended_progress.json의 base 단계) 추적 결과 CSV 없이도 DM-Count만 이어서 계산할
수 있다. 작업 디렉터리 전체, 영상, DM-Count 가중치를 옮긴 뒤 --work-dir/--video로 위치를
알려주면 된다. 체크포인트 서명은 경로가 아니라 영상 크기·설정·가중치 sha256을 보기 때문에
경로가 달라도 이어진다.

    python -m app.resume_extended <job_id> --device cuda:0 \\
        --dmcount-model-path models/dmcount_qnrf.pth --dmcount-gamma 1.5 --dmcount-input-scale 1.5 \\
        --pixels-per-meter 50 [--compute-only | --store-only] [--work-dir DIR] [--video PATH]

설정값(gamma/input-scale/pixels-per-meter)은 원래 worker 실행 때와 같아야 이어진다 — 다르면
서명이 맞지 않아 처음부터 다시 계산한다(조용히 섞인 결과를 만들지 않도록).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from uuid import UUID

import pandas as pd

from app.services.vision_results import BASE_DIR


def find_work_dir(base: Path, job_id: UUID) -> Path:
    """worker가 이 job에 쓴 attempt 디렉터리 중 core 파이프라인 결과가 있는 가장 최근 것."""
    root = base / 'uploads/analysis' / str(job_id)
    candidates = [p for p in root.glob('*/') if (p / 'roi_snapshot.json').is_file()
                  and ((p / 'pipeline').is_dir() or (p / 'extended').is_dir())]
    if not candidates:
        raise SystemExit(f'no worker work dir with roi_snapshot.json under {root}; pass --work-dir')
    return max(candidates, key=lambda p: p.stat().st_mtime)


def video_path_from_db(job_id: UUID, base: Path) -> Path:
    from sqlalchemy.orm import Session

    from app.database import engine
    from app.models import AnalysisJob, Video

    with Session(engine) as db:
        job = db.get(AnalysisJob, job_id)
        if job is None:
            raise SystemExit(f'analysis job {job_id} not found')
        return base / db.get(Video, job.video_id).storage_path


def compute(job_id: UUID, work: Path, video: Path, args) -> dict[str, Path]:
    from app.services.pipeline import run_extended_zone_pipeline
    from app.services.zone_analysis import load_zones_from_json

    zones = load_zones_from_json(str(work / 'roi_snapshot.json'))
    pipeline_dir = work / 'pipeline'
    core = {name: pipeline_dir / f'{name}.csv' for name in ('crowd_observations', 'flow', 'individual_directions')}
    have_core = all(p.is_file() for p in core.values())
    detections = work / 'tracker' / 'video_detections.csv'
    print(f'job {job_id}: resuming extended analysis in {work / "extended"} (video {video}, device {args.device})',
          flush=True)
    paths = run_extended_zone_pipeline(
        str(video), str(detections), zones, str(work / 'extended'),
        pixels_per_meter=args.pixels_per_meter,
        # core 결과가 있으면 재사용(특히 Optical Flow 재계산 방지). base 단계가 이미 끝났으면
        # 어차피 읽지 않으므로, 다른 컴퓨터로 extended/만 옮겨온 경우에도 문제없다.
        zone_timeseries=pd.read_csv(core['crowd_observations']) if have_core else None,
        motion_timeseries=pd.read_csv(core['flow']) if have_core else None,
        individual_directions=pd.read_csv(core['individual_directions']) if have_core else None,
        dmcount_model_path=args.dmcount_model_path,
        dmcount_device=args.device,
        dmcount_gamma=args.dmcount_gamma,
        dmcount_input_scale=args.dmcount_input_scale,
        resumable=True,
    )
    paths = {kind: Path(p).resolve() for kind, p in paths.items()}
    (work / 'extended' / 'resume_paths.json').write_text(
        json.dumps({k: v.relative_to(work.resolve()).as_posix() for k, v in paths.items()}, indent=2),
        encoding='utf-8')
    print('EXTENDED_COMPUTE_DONE', flush=True)
    return paths


def store(job_id: UUID, work: Path, paths: dict[str, Path] | None, pixels_per_meter: float | None) -> None:
    from sqlalchemy.orm import Session

    from app.database import engine
    from app.services import vision_extended_results as extended_storage

    if paths is None:
        record = work / 'extended' / 'resume_paths.json'
        if not record.is_file():
            raise SystemExit(f'{record} missing; run without --store-only first')
        paths = {k: (work / v).resolve() for k, v in json.loads(record.read_text(encoding='utf-8')).items()}
    if any(not p.is_relative_to(work.resolve()) for p in paths.values()):
        raise SystemExit('EXTENDED_PIPELINE_OUTPUT_PATH_NOT_ALLOWED')
    with Session(engine, expire_on_commit=False) as db:
        extended_storage.store_extended_result(db, job_id, paths, pixels_per_meter=pixels_per_meter)
    print(f'EXTENDED_STORED job {job_id}', flush=True)


def main(argv: list[str] | None = None) -> int:
    from app.services.inference_runtime import normalize_device

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('job_id', type=UUID)
    parser.add_argument('--work-dir', type=Path, help='worker attempt dir (default: newest under uploads/analysis/<job>)')
    parser.add_argument('--video', type=Path, help='source video (default: looked up in the DB)')
    parser.add_argument('--device', type=normalize_device, default='cpu', help='cpu, cuda:N or GPU index')
    parser.add_argument('--pixels-per-meter', type=float, default=None)
    parser.add_argument('--dmcount-model-path', type=Path, default=None)
    parser.add_argument('--dmcount-gamma', type=float, default=1.0)
    parser.add_argument('--dmcount-input-scale', type=float, default=1.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--compute-only', action='store_true', help='compute without touching the DB (needs --video)')
    mode.add_argument('--store-only', action='store_true', help='store an already computed result')
    args = parser.parse_args(argv)
    for name in ('dmcount_gamma', 'dmcount_input_scale'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'{name.replace("_", "-")} must be positive and finite')
    if args.pixels_per_meter is not None and args.pixels_per_meter <= 0:
        parser.error('pixels-per-meter must be positive')
    if args.dmcount_model_path is not None and (not args.dmcount_model_path.is_file()
                                                or args.dmcount_model_path.stat().st_size == 0):
        parser.error('dmcount-model-path must point to an existing local checkpoint; no automatic download')
    if args.compute_only and args.video is None:
        parser.error('--compute-only needs --video (it does not read the DB)')

    base = Path(BASE_DIR).resolve()
    work = (args.work_dir or find_work_dir(base, args.job_id)).resolve()
    paths = None
    if not args.store_only:
        video = (args.video or video_path_from_db(args.job_id, base)).resolve()
        if not video.is_file():
            parser.error(f'video not found: {video}')
        paths = compute(args.job_id, work, video, args)
    if not args.compute_only:
        store(args.job_id, work, paths, args.pixels_per_meter)
    return 0


if __name__ == '__main__':
    sys.exit(main())
