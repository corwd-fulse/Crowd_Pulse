"""영상 하나를 분석해서 결과 CSV들을 만드는 단일 진입점.

FastAPI 라우터나 CLI에서 이 함수 하나만 호출하면 구역 통계·Optical Flow·개인별
이동 방향·병목 점수를 전부 계산해 out_dir에 CSV로 저장한다.
"""

from __future__ import annotations

from pathlib import Path
import json
import time

import cv2
import pandas as pd

from .bottleneck import compute_bottleneck_scores
from .individual_flow import aggregate_by_zone, compute_individual_directions
from .optical_flow import compute_zone_motion
from .zone_analysis import Zone, compute_zone_timeseries
from .hybrid_flow import build_hybrid_summary
from .bottleneck_diagnostics import describe_bottleneck_windows


def _run_zone_pipeline(
    video_path: str,
    detections_csv: str,
    zones: list[Zone],
    out_dir: str,
    *, analysis_profile: str = 'full',
    flow_method: str = 'full_frame',
    _stack=None,
) -> dict[str, str]:
    """반환: {결과종류: 저장된 csv 경로}.

    video_path: optical_flow 계산에 쓰는 원본 영상(detections.csv 좌표만으론 안 됨).
    detections_csv: tracker.py가 만든 frame,t_ms,track_id,x1,y1,x2,y2,conf 형식 csv.
    """
    if analysis_profile not in ('fast', 'full'):
        raise ValueError('Invalid analysis profile')
    if flow_method not in ('full_frame', 'roi_crop_128'):
        raise ValueError('Invalid flow method')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width,height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    started = time.perf_counter()
    streamed = Path(detections_csv).is_file() and Path(detections_csv).stat().st_size > 8*1024**2
    if streamed:
        from .streaming_tracks import process
        stack=_stack
        zone_ts,individual_zone_agg,prepared_votes=stack.enter_context(process(detections_csv,zones,fps,out / "individual_directions.csv"))
    else:
        zone_ts = compute_zone_timeseries(detections_csv, zones, fps=fps)
    flow_started = time.perf_counter()
    motion_ts = (compute_zone_motion(video_path, zones, fps=fps, flow_method=flow_method) if analysis_profile == 'full'
                 else pd.DataFrame(columns=['sec','zone','angle_deg','magnitude_px','motion','low_confidence']))
    flow_seconds = time.perf_counter() - flow_started
    if not streamed:
        individual = compute_individual_directions(detections_csv)
        individual_zone_agg = aggregate_by_zone(individual, detections_csv, zones, fps=fps)

    if analysis_profile == 'full':
        merged = zone_ts.merge(motion_ts[["sec", "zone", "motion"]], on=["sec", "zone"], how="left")
        scores = compute_bottleneck_scores(merged, missing_policy='omit_unavailable_v1')
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
    if not streamed:individual.to_csv(paths["individual_directions"], index=False)
    individual_zone_agg.to_csv(paths["individual_zone_agg"], index=False)
    scores.to_csv(paths["bottleneck_scores"], index=False)

    try:
        hybrid = build_hybrid_summary(None if streamed else individual, detections_csv, zones, motion_ts, zone_ts,
                                      **({"prepared_votes":prepared_votes} if streamed else {}))
    finally:
        if streamed:stack.close()
    score_keys = {(int(r.window_start), r.zone) for r in scores.itertuples()}
    diagnostic_rows = zone_ts.merge(motion_ts[['sec', 'zone', 'motion', 'low_confidence']],
                                    on=['sec', 'zone'], how='left').to_dict(orient='records')
    hybrid_rows = {(r['sec'], r['zone']): r for r in hybrid.get('rows', [])}
    for row in diagnostic_rows:
        row['hybrid'] = hybrid_rows.get((row['sec'], row['zone']))
    hybrid['bottleneck_status'] = describe_bottleneck_windows(
        diagnostic_rows, score_keys, analysis_profile=analysis_profile)
    metadata = dict(flow_method=flow_method if analysis_profile == 'full' else 'not_run',
                    requested_flow_method=flow_method, crop_padding_px=128 if flow_method == 'roi_crop_128' else None,
                    bottleneck_formula='legacy-weights-v1', bottleneck_missing_policy='omit_unavailable_v1',
                    individual_speed_px_semantics='displacement over observation window; not px/frame or m/s',
                    farneback_parameters=dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                             poly_n=5, poly_sigma=1.2, flags=0),
                    flow_seconds=flow_seconds, zone_pipeline_seconds=time.perf_counter() - started,
                    hybrid=hybrid)
    if Path(detections_csv).is_file():
        from .transition_artifact import summarize_tracking
        metadata['transitions'] = summarize_tracking(detections_csv,zones,fps,frame_count,width,height)
        (out/'zone_transitions.json').write_text(json.dumps(metadata['transitions'],allow_nan=False),encoding='utf-8')
    # Sidecar is copied into execution_snapshot.settings by the Worker. The five
    # existing CSV headers and result manifest remain exactly schema v1.
    (out / 'analysis_details.json').write_text(json.dumps(metadata, allow_nan=False), encoding='utf-8')

    return {k: str(v) for k, v in paths.items()}


def run_zone_pipeline(video_path, detections_csv, zones, out_dir, *, analysis_profile="full", flow_method="full_frame"):
    from contextlib import ExitStack
    with ExitStack() as stack:
        return _run_zone_pipeline(video_path,detections_csv,zones,out_dir,analysis_profile=analysis_profile,flow_method=flow_method,_stack=stack)
