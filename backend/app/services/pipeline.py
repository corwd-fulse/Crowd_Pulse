"""영상 하나를 분석해서 결과 CSV들을 만드는 단일 진입점.

FastAPI 라우터나 CLI에서 이 함수 하나만 호출하면 구역 통계·Optical Flow·개인별
이동 방향·병목 점수를 전부 계산해 out_dir에 CSV로 저장한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd

from .bottleneck import compute_bottleneck_scores
from .density_heatmap import compute_density_heatmap
from .individual_flow import aggregate_by_zone, compute_counter_flow, compute_individual_directions
from .optical_flow import compute_zone_motion
from .queue_analysis import compute_queue_scores
from .tracking_eval import evaluate_tracking
from .zone_analysis import Zone, compute_dwell_times, compute_zone_timeseries


def run_zone_pipeline(
    video_path: str,
    detections_csv: str,
    zones: list[Zone],
    out_dir: str,
    *, analysis_profile: str = 'full',
) -> dict[str, str]:
    """반환: {결과종류: 저장된 csv 경로}.

    video_path: optical_flow 계산에 쓰는 원본 영상(detections.csv 좌표만으론 안 됨).
    detections_csv: tracker.py가 만든 frame,t_ms,track_id,x1,y1,x2,y2,conf 형식 csv.
    """
    if analysis_profile not in ('fast', 'full'):
        raise ValueError('Invalid analysis profile')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    finally:
        cap.release()

    zone_ts = compute_zone_timeseries(detections_csv, zones, fps=fps)
    motion_ts = (compute_zone_motion(video_path, zones, fps=fps) if analysis_profile == 'full'
                 else pd.DataFrame(columns=['sec','zone','angle_deg','magnitude_px','motion','low_confidence']))
    individual = compute_individual_directions(detections_csv)
    individual_zone_agg = aggregate_by_zone(individual, detections_csv, zones, fps=fps)

    if analysis_profile == 'full':
        merged = zone_ts.merge(motion_ts[["sec", "zone", "motion"]], on=["sec", "zone"], how="left")
        scores = compute_bottleneck_scores(merged)
    else:
        scores = pd.DataFrame(columns=['window_start','zone','score','growth','slowdown','imbalance','risk_level'])

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


def run_extended_zone_pipeline(
    video_path: str,
    detections_csv: str,
    zones: list[Zone],
    out_dir: str,
) -> dict[str, str]:
    """밀집 히트맵·체류 시간·역방향 비율·대기열 점수·추적 품질 평가, 별도 CSV 산출.

    run_zone_pipeline()의 다섯 결과 파일(crowd_observations/flow/individual_directions/
    individual_zone_agg/bottleneck_scores)은 worker.py가 analysis_jobs DB 저장 시 정확히
    이 다섯 개 키만 있는지 검증한다(app/worker.py의 set(paths)!=set(SCHEMAS) 체크).
    그 계약을 깨지 않도록 이 함수는 완전히 별도의 파일/딕셔너리로 결과를 낸다 — 아직 DB
    테이블/마이그레이션은 없고, Forecast 쪽에서 쓸 수 있는 CSV만 만든다.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    dwell = compute_dwell_times(detections_csv, zones, fps=fps)
    individual = compute_individual_directions(detections_csv)
    counter_flow = compute_counter_flow(individual, detections_csv, zones, fps=fps)

    zone_ts = compute_zone_timeseries(detections_csv, zones, fps=fps)
    motion_ts = compute_zone_motion(video_path, zones, fps=fps)
    merged = zone_ts.merge(motion_ts[["sec", "zone", "motion"]], on=["sec", "zone"], how="left")
    queue_scores = compute_queue_scores(merged)

    heatmap = compute_density_heatmap(detections_csv, width, height, fps=fps)
    tracking_quality = evaluate_tracking(detections_csv)

    paths = {
        "dwell_times": out / "dwell_times.csv",
        "counter_flow": out / "counter_flow.csv",
        "queue_scores": out / "queue_scores.csv",
        "density_heatmap": out / "density_heatmap.csv",
        "tracking_quality": out / "tracking_quality.json",
    }
    dwell.to_csv(paths["dwell_times"], index=False)
    counter_flow.to_csv(paths["counter_flow"], index=False)
    queue_scores.to_csv(paths["queue_scores"], index=False)
    heatmap.to_csv(paths["density_heatmap"], index=False)
    paths["tracking_quality"].write_text(json.dumps(tracking_quality, indent=2, ensure_ascii=False))

    return {k: str(v) for k, v in paths.items()}
