"""로컬 영상 1개에 대해 구역 통계·Optical Flow·개인별 이동 방향·병목 점수를 한 번에 돌리는 CLI.

FastAPI 라우터에서는 CLI 대신 app.services.pipeline.run_zone_pipeline()을 직접 호출하면 된다.

사용 예:
  python scripts/analyze_zones.py \
      --video /path/to/MOT20-01_bytetrack.mp4 \
      --detections /path/to/MOT20-01_detections.csv \
      --zone "A:0,0;640,0;640,480;0,480" \
      --out-dir out/MOT20-01

ROI는 "이름:x1,y1;x2,y2;..." 형식으로 여러 번 줄 수 있다(픽셀 좌표, 꼭짓점 >= 3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/ 를 import 경로에 추가

import pandas as pd

from app.services.pipeline import run_zone_pipeline
from app.services.zone_analysis import Zone, load_zones_from_json


def _parse_zone(spec: str) -> Zone:
    name, _, pts = spec.partition(":")
    polygon = [tuple(map(float, p.split(","))) for p in pts.split(";")]
    return Zone(name=name, polygon=polygon)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--zone", action="append", dest="zones", help="\"이름:x,y;x,y;...\" (여러 번 지정 가능)")
    parser.add_argument("--zones-file", help="configs/zones/*.json 경로 (--zone 대신 사용)")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    if not args.zones and not args.zones_file:
        parser.error("--zone 또는 --zones-file 중 하나는 필요합니다")

    zones = [_parse_zone(z) for z in args.zones] if args.zones else load_zones_from_json(args.zones_file)
    paths = run_zone_pipeline(args.video, args.detections, zones, args.out_dir)

    print("완료:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")

    scores = pd.read_csv(paths["bottleneck_scores"])
    if not scores.empty:
        print()
        print(scores.sort_values("score", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
