"""영상 하나를 분석해서 결과 CSV들을 만드는 단일 진입점.

FastAPI 라우터나 CLI에서 이 함수 하나만 호출하면 구역 통계·Optical Flow·개인별
이동 방향·병목 점수를 전부 계산해 out_dir에 CSV로 저장한다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from .bottleneck import compute_bottleneck_scores
from .individual_flow import aggregate_by_zone, compute_individual_directions
from .optical_flow import compute_zone_motion
from .zone_analysis import Zone, compute_zone_timeseries


def run_zone_pipeline(
    video_path: str,
    detections_csv: str,
    zones: list[Zone],
    out_dir: str,
) -> dict[str, str]:
    """반환: {결과종류: 저장된 csv 경로}.

    video_path: optical_flow 계산에 쓰는 원본 영상(detections.csv 좌표만으론 안 됨).
    detections_csv: tracker.py가 만든 frame,t_ms,track_id,x1,y1,x2,y2,conf 형식 csv.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS) or 30.0

    zone_ts = compute_zone_timeseries(detections_csv, zones, fps=fps)
    motion_ts = compute_zone_motion(video_path, zones, fps=fps)
    individual = compute_individual_directions(detections_csv)
    individual_zone_agg = aggregate_by_zone(individual, detections_csv, zones, fps=fps)

    merged = zone_ts.merge(motion_ts[["sec", "zone", "motion"]], on=["sec", "zone"], how="left")
    scores = compute_bottleneck_scores(merged)

    paths = {
        "crowd_observations": out / "crowd_observations.csv",
        "flow": out / "flow.csv",
        "individual_directions": out / "individual_directions.csv",
        "individual_zone_agg": out / "individual_zone_agg.csv",
        "bottleneck_scores": out / "bottleneck_scores.csv",
    }
    zone_ts.to_csv(paths["crowd_observations"], index=False)
    motion_ts.to_csv(paths["flow"], index=False)
    individual.to_csv(paths["individual_directions"], index=False)
    individual_zone_agg.to_csv(paths["individual_zone_agg"], index=False)
    scores.to_csv(paths["bottleneck_scores"], index=False)

    return {k: str(v) for k, v in paths.items()}
