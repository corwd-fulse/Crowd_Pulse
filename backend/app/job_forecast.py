"""끝난 분석 job 하나에 시계열 예측을 돌리고 DB에 저장한다(worker --auto-forecast 와 같은 동작, 수동 재실행용).

python -m app.job_forecast <job_id> <job 출력 폴더> [--forecast-python PATH] [--pre-dir DIR]
  job 출력 폴더 = uploads/analysis/<job_id>/<run>/ (pipeline/, tracker/, extended/, roi_snapshot.json 이 있는 곳)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('job_id', type=UUID)
    ap.add_argument('output_dir', type=Path)
    ap.add_argument('--forecast-python', default=None, help='lightgbm·torch가 있는 파이썬(기본: 지금 파이썬)')
    ap.add_argument('--pre-dir', type=Path, default=None, help='미리 학습 모델 폴더(기본 forecast/serving/artifacts/video)')
    a = ap.parse_args(argv)
    from app.database import SessionLocal
    from app.services.job_forecast import run_for_job

    report = run_for_job(SessionLocal, a.job_id, a.output_dir, python=a.forecast_python, pre_dir=a.pre_dir)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['sources'] and all('error' not in r for r in report['sources'].values()) else 1


if __name__ == '__main__':
    sys.exit(main())
