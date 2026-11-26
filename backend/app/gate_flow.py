"""카메라 한 대의 경계선 통과 수(방향별, 초별)를 계산해 uploads/results/<job_id>/gate_flow.json 에 저장한다.

python -m app.gate_flow <job_id> <분석 출력 폴더> --axis x|y [--line 0~1]
  axis x: 화면 세로선을 왼쪽(minus)/오른쪽(plus)으로 지나감. axis y: 화면 가로선을 위로(minus, 카메라에서 멀어짐)/아래로(plus).
  line을 주지 않으면 ROI 구역의 가운데.
"""
import argparse
import json
import sys
from pathlib import Path

from app.services.gate_flow import compute_for_job_dir, write
from app.services.vision_results import BASE_DIR


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('job_id'); ap.add_argument('out_dir', type=Path)
    ap.add_argument('--axis', choices=('x', 'y'), required=True)
    ap.add_argument('--line', type=float, default=None)
    a = ap.parse_args(argv)
    counts, meta = compute_for_job_dir(a.out_dir, a.axis, a.line)
    dest = BASE_DIR / 'uploads/results' / a.job_id
    dest.mkdir(parents=True, exist_ok=True)
    p = write(dest, a.job_id, counts, meta)
    print(json.dumps({'file': str(p), 'seconds': len(counts), 'plus': int(counts.plus.sum()), 'minus': int(counts.minus.sum()), **meta}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
