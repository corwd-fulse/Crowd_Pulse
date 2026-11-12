"""ByteTrack 트랙 궤적 기반 개인별 이동 방향 계산.

FR-ZON-03(구역 평균 Optical Flow)의 보완 기능이다. 스펙(FR-ZON-01~05)에는 없는
추가 기능이라 팀 합의가 필요하다 — "구역 평균"은 optical_flow.py가 그대로 맡고,
이 모듈은 ByteTrack이 이미 갖고 있는 track_id별 좌표 이력으로 "그 사람 개인의"
이동 방향만 추가로 낸다.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .zone_analysis import OUTSIDE, Zone, _bottom_center, _has_id_switch, _zone_of_point

# 화살표 길이를 속도(px)에 비례시킬 때 쓰는 배율/최소·최대 길이(px). 거의 안 움직이는
# 사람도 구분은 되게 최소 길이를 주고, 너무 길어져 화면을 덮지 않게 상한을 둔다.
ARROW_SPEED_SCALE = 3.0
ARROW_MIN_LEN = 4
ARROW_MAX_LEN = 80

# 이 프레임 수만큼 이전 위치와 비교해 방향을 구한다. 프레임마다 비교하면 박스 떨림이
# 그대로 방향에 묻어나오므로, zone_analysis의 떨림/스위칭 조사에서 쓴 것과 같은 크기의
# 창으로 묶어 "최근 실제 이동"을 본다.
DEFAULT_WINDOW_FRAMES = 5


def compute_individual_directions(
    detections_csv: str,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
) -> pd.DataFrame:
    """반환: columns = [frame, t_ms, track_id, angle_deg, speed_px, reliable].

    reliable=False인 행은 ID 스위칭 의심 구간이라 angle_deg/speed_px가 None이다.
    """
    det = pd.read_csv(detections_csv)
    rows = []

    for tid, g in det.sort_values("frame").groupby("track_id"):
        g = g.reset_index(drop=True)
        points = list(zip(((g.x1 + g.x2) / 2).tolist(), g.y2.tolist()))
        frames = g["frame"].tolist()
        t_ms = g["t_ms"].tolist()

        for i in range(len(points)):
            lo = max(0, i - window_frames)
            if lo == i:
                continue  # 비교할 이전 지점이 아직 없음
            reliable = not _has_id_switch(points, i)
            angle = speed = None
            if reliable:
                dx = points[i][0] - points[lo][0]
                dy = points[i][1] - points[lo][1]
                speed = (dx**2 + dy**2) ** 0.5
                if speed > 0:
                    angle = float(np.degrees(np.arctan2(dy, dx)) % 360)
            rows.append(
                {
                    "frame": frames[i],
                    "t_ms": t_ms[i],
                    "track_id": tid,
                    "angle_deg": angle,
                    "speed_px": speed,
                    "reliable": reliable,
                }
            )

    return pd.DataFrame(rows, columns=["frame", "t_ms", "track_id", "angle_deg", "speed_px", "reliable"])


def aggregate_by_zone(
    individual: pd.DataFrame,
    detections_csv: str,
    zones: list[Zone],
    fps: float,
) -> pd.DataFrame:
    """개인별 방향을 구역·1초 버킷으로 묶어 평균 방향(원형 평균)을 낸다.

    optical_flow.compute_zone_motion()의 angle_deg와 같은 구간을 비교해 두 방법이
    서로 맞는지 교차검증하는 데 쓴다.
    """
    det = pd.read_csv(detections_csv)
    det["zone"] = [
        _zone_of_point(_bottom_center(x1, y1, x2, y2), zones)
        for x1, y1, x2, y2 in zip(det.x1, det.y1, det.x2, det.y2)
    ]

    merged = individual.merge(det[["frame", "track_id", "zone"]], on=["frame", "track_id"], how="left")
    merged = merged[(merged.zone != OUTSIDE) & merged.reliable & merged.angle_deg.notna()].copy()
    merged["sec"] = (merged.frame / fps).astype(int)

    rows = []
    for (sec, zone), g in merged.groupby(["sec", "zone"]):
        rad = np.radians(g.angle_deg.astype(float))
        mean_angle = float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)
        rows.append({"sec": sec, "zone": zone, "mean_angle_deg": mean_angle, "n_tracks": len(g)})

    return pd.DataFrame(rows, columns=["sec", "zone", "mean_angle_deg", "n_tracks"])


def draw_direction_arrows(
    frame: np.ndarray,
    positions: dict[int, tuple[float, float]],
    directions: pd.DataFrame,
) -> np.ndarray:
    """프레임 위에 track_id별 이동 방향 화살표를 그린다. 길이는 속도(px)에 비례한다.

    positions: {track_id: (cx, cy)} 그 프레임의 위치. directions: compute_individual_directions() 출력.
    신뢰 불가(reliable=False) 트랙은 화살표 대신 보라색 원으로 표시한다.
    """
    dirs = directions.set_index("track_id")
    for tid, (cx, cy) in positions.items():
        if tid not in dirs.index:
            continue
        d = dirs.loc[tid]
        if not d["reliable"]:
            cv2.circle(frame, (int(cx), int(cy)), 8, (255, 0, 255), 2)
            continue
        if pd.isna(d["angle_deg"]):
            continue
        length = min(max(d["speed_px"] * ARROW_SPEED_SCALE, ARROW_MIN_LEN), ARROW_MAX_LEN)
        angle = np.radians(d["angle_deg"])
        ex = int(cx + length * np.cos(angle))
        ey = int(cy + length * np.sin(angle))
        cv2.arrowedLine(frame, (int(cx), int(cy)), (ex, ey), (0, 0, 255), 2, tipLength=0.3)
    return frame
