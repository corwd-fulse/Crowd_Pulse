"""Optical Flow 기반 구역 평균 이동량 계산 (FR-ZON-03).

Farneback 알고리즘으로 연속 프레임 간 이동 벡터를 구하고, 구역 마스크 안
벡터를 평균해 angle_deg/magnitude_px/mean_motion(정규화)으로 요약한다.
m/s 로는 절대 환산하지 않는다(카메라 보정 전 픽셀 단위 보조 지표).
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .zone_analysis import Zone

MIN_VECTORS = 25  # 구역이 너무 작아 벡터 수 부족으로 보는 임계값 (shortcut: 임의값, 실footage로 재조정 필요)
SCENE_CHANGE_DIFF = 40.0  # shortcut: 프레임 평균 밝기 변화가 이 값을 넘으면 장면전환/흔들림으로 간주한 휴리스틱


def _zone_mask(zone: Zone, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array(zone.polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def compute_zone_motion(video_path: str, zones: list[Zone], fps: float | None = None) -> pd.DataFrame:
    """반환: columns = [sec, zone, angle_deg, magnitude_px, motion, low_confidence].

    주의: 각도를 프레임마다 구한 뒤 "각도끼리" 평균 내면 0/360도 경계(예: 358도와 2도)에서
    산술평균이 180도처럼 완전히 틀린 값이 나온다(원형 통계 문제). 그래서 프레임별 각도/크기로
    바꾸기 전의 (mx, my) 벡터 성분 상태로 누적해서 평균 낸 뒤, 마지막에 한 번만 각도/크기로
    변환한다 — 벡터 합산은 경계를 지나도 불연속이 없다.
    """
    cap = cv2.VideoCapture(video_path)
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    diag = (w**2 + h**2) ** 0.5  # shortcut: motion 정규화 기준 = 프레임 대각선 길이 (카메라 보정 전 임시 기준)

    masks = {z.name: _zone_mask(z, (h, w)) for z in zones}

    ok, prev = cap.read()
    columns = ["sec", "zone", "angle_deg", "magnitude_px", "motion", "low_confidence"]
    if not ok:
        cap.release()
        return pd.DataFrame(columns=columns)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        scene_change = float(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16)).mean()) > SCENE_CHANGE_DIFF
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev_gray = gray
        sec = frame_idx // int(round(fps))

        for name, mask in masks.items():
            if scene_change:
                rows.append((sec, name, None, None, True))
                continue
            fx = flow[..., 0][mask == 1]
            fy = flow[..., 1][mask == 1]
            if fx.size < MIN_VECTORS:
                rows.append((sec, name, None, None, True))
                continue
            rows.append((sec, name, float(fx.mean()), float(fy.mean()), False))
    cap.release()

    df = pd.DataFrame(rows, columns=["sec", "zone", "mx", "my", "low_confidence"])
    agg = (
        df.groupby(["sec", "zone"], as_index=False)
        .agg(mx=("mx", "mean"), my=("my", "mean"), low_confidence=("low_confidence", "any"))
    )
    magnitude = np.sqrt(agg["mx"] ** 2 + agg["my"] ** 2)
    agg["magnitude_px"] = magnitude
    agg["motion"] = magnitude / diag * 100
    agg["angle_deg"] = np.degrees(np.arctan2(agg["my"], agg["mx"])) % 360
    agg.loc[agg["low_confidence"], ["angle_deg", "magnitude_px", "motion"]] = None
    return agg[columns]
