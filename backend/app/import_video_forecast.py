"""영상 시계열 예측 결과 파일을 DB(crowd_predictions + video_forecast_runs)로 옮긴다.

python -m app.import_video_forecast <job_id> [<job_id> ...] [--source yolo dmcount fused] [--root DIR]
python -m app.import_video_forecast --all      uploads/results/*/forecast 아래 결과가 있는 job 전부

- 같은 job과 출처를 다시 넣으면 이전 행을 지우고 교체한다(한 job x 출처 = 한 트랜잭션, 중복 없음).
- 파일은 지우지 않는다. 파일만 있고 DB에 없는 결과도 API가 계속 파일에서 읽는다.
- analysis_jobs에 없는 job은 건너뛴다(외래 키).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

from app.services import video_forecast_results as svc


def _jobs_with_files(root: Path) -> list[UUID]:
    out = []
    for d in sorted(root.glob('*/forecast')):
        try:
            job = UUID(d.parent.name)
        except ValueError:
            continue
        if any((d / s / 'summary.json').is_file() for s in svc.SOURCES):
            out.append(job)
    return out


def _provenance(folder: Path) -> dict:
    """미리 학습 모델 폴더(--pre-dir로 돌렸으면 pre_loo)의 버전을 남긴다."""
    info = {}
    loo = folder / 'pre_loo' / 'manifest.json'
    if loo.is_file():
        m = json.loads(loo.read_text(encoding='utf-8')).get('models', {})
        info['pretrained'] = {'dir': 'pre_loo (trained without this video)',
                              'versions': sorted({v.get('version') for v in m.values() if v.get('version')})}
    return info


def import_job(db, job_id: UUID, sources, root: Path | None) -> list[dict]:
    folder = Path(root) / str(job_id) / 'forecast' if root else svc.forecast_root(job_id)
    done = []
    for source in sources:
        if not (folder / source / 'summary.json').is_file():
            continue
        try:
            done.append(svc.import_run(db, job_id, source, folder, _provenance(folder)))
            db.commit()
        except Exception:
            db.rollback()
            raise
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('jobs', nargs='*', type=UUID)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--source', nargs='+', default=list(svc.SOURCES), choices=svc.SOURCES)
    ap.add_argument('--root', type=Path, help='results 폴더(기본 uploads/results)')
    a = ap.parse_args(argv)
    from app.database import SessionLocal
    from app.models.analytics import AnalysisJob

    root = a.root or (svc.forecast_root(UUID(int=0)).parent.parent)
    jobs = _jobs_with_files(root) if a.all else a.jobs
    if not jobs:
        ap.error('job_id를 주거나 --all을 쓰세요')
    with SessionLocal() as db:
        for job in jobs:
            if db.get(AnalysisJob, job) is None:
                print(json.dumps({'job_id': str(job), 'skipped': 'analysis job not in DB'}))
                continue
            for r in import_job(db, job, a.source, a.root):
                print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
