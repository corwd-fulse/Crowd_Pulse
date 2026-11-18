"""YOLO/ByteTrack 추적 품질 평가 (정답 레이블 없는 운영 환경용 대체 지표).

idf1/mota 같은 정식 MOT 지표는 프레임별 정답(ground truth) 박스가 있어야 계산할 수 있고,
그건 tracker.py 주석에 적힌 대로 모델 선정 시 Colab에서 MOT20으로 1회 비교했을 뿐 운영
파이프라인엔 없다. 실제 CCTV 영상엔 정답이 없으므로, 이미 가진 tracker.py 출력(detections.csv)
만으로 "이 추적이 얼마나 안정적이었는지" 가늠하는 대체 지표를 낸다 — 트랙 생존 길이, ID
스위칭 의심 비율, 탐지 신뢰도 분포. 정답 기반 정확도가 아니라 안정성 지표임을 분명히 한다.
"""

from __future__ import annotations

import pandas as pd

from .zone_analysis import ID_SWITCH_WINDOW, _has_id_switch


def evaluate_tracking(detections_csv: str) -> dict:
    """반환: 트랙 생존 길이(프레임)/신뢰도 통계 + ID 스위칭 의심 비율."""
    det = pd.read_csv(detections_csv)
    if det.empty:
        return {
            "track_count": 0,
            "mean_track_length_frames": 0.0,
            "median_track_length_frames": 0.0,
            "mean_confidence": 0.0,
            "min_confidence": 0.0,
            "suspected_id_switch_ratio": 0.0,
        }

    lengths = det.groupby("track_id").size()
    suspect_tracks = 0
    for _track_id, g in det.sort_values("frame").groupby("track_id"):
        points = list(zip(((g.x1 + g.x2) / 2).tolist(), g.y2.tolist()))
        if any(_has_id_switch(points, i, window=ID_SWITCH_WINDOW) for i in range(len(points))):
            suspect_tracks += 1

    return {
        "track_count": int(lengths.shape[0]),
        "mean_track_length_frames": round(float(lengths.mean()), 1),
        "median_track_length_frames": float(lengths.median()),
        "mean_confidence": round(float(det.conf.mean()), 3),
        "min_confidence": round(float(det.conf.min()), 3),
        "suspected_id_switch_ratio": round(suspect_tracks / lengths.shape[0], 3),
    }
