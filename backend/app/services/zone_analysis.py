"""구역(ROI) 별 인원 수·유입/유출 계산 (FR-ZON-02).

tracker.py 가 뽑은 detections.csv(frame,t_ms,track_id,x1,y1,x2,y2,conf)와
ROI 다각형을 받아 1초 버킷으로 zone_count/inflow/outflow를 계산한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

MIN_DWELL_FRAMES = 3  # 경계 떨림으로 인한 중복 유입/유출 방지 (스펙 제안값)

# 체류 프레임 조건을 만족해도, 후보 상태가 시작된 지점부터 confirmation 지점까지
# 실제로 이 거리(px) 이상 움직이지 않으면 "진짜 횡단"으로 보지 않는다. 경계선
# 바로 위에 서 있는 사람의 박스 떨림이 체류 프레임 필터를 통과해버리는 걸 막는다.
# 실데이터(E05_001)에서 경계 떨림 사례의 net 이동량은 최대 11.9px, 진짜 횡단은
# 최소 20px로 갈라져 있어 그 사이값으로 잡았다.
MIN_NET_DISPLACEMENT_PX = 15.0

# ByteTrack이 사람이 겹칠 때(occlusion) 같은 track_id로 다른 사람을 가리키게 되면
# 박스 위치가 순간적으로 크게 튄다. confirmation 시점 주변에서 그 트랙의 평소
# 프레임간 이동량 대비 비정상적으로 큰 점프가 있으면 ID 스위칭으로 의심하고
# 그 횡단 이벤트는 집계에서 제외한다(신뢰 불가 데이터이므로 추측 대신 드롭).
ID_SWITCH_JUMP_PX = 40.0
ID_SWITCH_JUMP_RATIO = 6.0
ID_SWITCH_WINDOW = 15

# "구역 밖" 상태의 센티널. pandas DataFrame 컬럼에 None을 섞으면 NaN으로 바뀌어
# nan == nan이 항상 False가 되므로(떨림 필터의 상태 비교가 깨짐), 명시적 문자열을 쓴다.
OUTSIDE = ""


@dataclass
class Zone:
    name: str
    polygon: list[tuple[float, float]]  # pixel 좌표 [[x,y], ...], 꼭짓점 >= 3


def _bottom_center(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    return ((x1 + x2) / 2, y2)


def _zone_of_point(point: tuple[float, float], zones: list[Zone]) -> str:
    for zone in zones:
        contour = np.array(zone.polygon, dtype=np.float32)
        if cv2.pointPolygonTest(contour, point, False) >= 0:
            return zone.name
    return OUTSIDE


def _dist(p: tuple[float, float], q: tuple[float, float]) -> float:
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def _has_id_switch(points: list[tuple[float, float]], idx: int, window: int = ID_SWITCH_WINDOW) -> bool:
    """idx 프레임 주변에서, 그 트랙 평소 이동량 대비 비정상적으로 큰 순간 점프가 있는지 본다."""
    lo, hi = max(0, idx - window), min(len(points), idx + window + 1)
    pts = points[lo:hi]
    if len(pts) < 3:
        return False
    steps = [_dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    steps.sort()
    median = steps[len(steps) // 2]
    max_step = steps[-1]
    return max_step > ID_SWITCH_JUMP_PX and max_step > ID_SWITCH_JUMP_RATIO * max(median, 1e-6)


def _confirmed_transitions(
    frames: list[int],
    raw_states: list[str],
    points: list[tuple[float, float]],
    min_dwell_frames: int,
    min_net_displacement: float = MIN_NET_DISPLACEMENT_PX,
) -> list[tuple[int, str, str]]:
    """프레임별 원시 구역 상태열에서, 체류 프레임·순이동거리 조건을 만족하고 ID
    스위칭 의심이 없는 전환만 (frame, from, to)로 반환한다."""
    confirmed = raw_states[0]
    candidate: str | None = None  # None = "아직 후보 없음" (OUTSIDE=""와 구분됨)
    candidate_run = 0
    candidate_start_idx: int | None = None
    events: list[tuple[int, str, str]] = []

    for i in range(1, len(frames)):
        frame, raw = frames[i], raw_states[i]
        if raw == confirmed:
            candidate, candidate_run, candidate_start_idx = None, 0, None
            continue
        if raw == candidate:
            candidate_run += 1
        else:
            candidate, candidate_run, candidate_start_idx = raw, 1, i

        if candidate_run >= min_dwell_frames and candidate != confirmed:
            displacement = _dist(points[candidate_start_idx], points[i])
            if displacement < min_net_displacement:
                continue  # 체류는 했지만 실제로는 거의 안 움직임 -> 더 지켜본다

            if _has_id_switch(points, i):
                # 신뢰할 수 없는 구간이라 이 전환은 세지 않되, 추적 상태는 갱신해서
                # 같은 위치에서 재확인이 반복되지 않게 한다.
                confirmed, candidate, candidate_run, candidate_start_idx = candidate, None, 0, None
                continue

            events.append((frame, confirmed, candidate))
            confirmed, candidate, candidate_run, candidate_start_idx = candidate, None, 0, None

    return events


def compute_zone_timeseries(
    detections_csv: str,
    zones: list[Zone],
    fps: float,
    min_dwell_frames: int = MIN_DWELL_FRAMES,
) -> pd.DataFrame:
    """반환: columns = [sec, zone, count, inflow, outflow]. tracker 데이터가 없으면 inflow/outflow는 NaN."""
    det = pd.read_csv(detections_csv)
    zone_names = [z.name for z in zones]

    det["zone"] = [
        _zone_of_point(_bottom_center(x1, y1, x2, y2), zones)
        for x1, y1, x2, y2 in zip(det.x1, det.y1, det.x2, det.y2)
    ]
    det["sec"] = det.t_ms // 1000
    max_sec = int(det.sec.max()) if len(det) else 0

    # count: 구역에 있던 프레임 수를 1초 버킷 평균 (빈 구역/빈 초도 0으로 채움)
    per_frame = det[det.zone != OUTSIDE].groupby(["frame", "sec", "zone"]).size().reset_index(name="n")
    counts = per_frame.groupby(["sec", "zone"])["n"].mean().round(1)
    full_index = pd.MultiIndex.from_product([range(max_sec + 1), zone_names], names=["sec", "zone"])
    counts = counts.reindex(full_index, fill_value=0.0).reset_index(name="count")

    has_track_id = det["track_id"].notna().any()
    if not has_track_id:
        counts["inflow"] = float("nan")
        counts["outflow"] = float("nan")
        return counts[["sec", "zone", "count", "inflow", "outflow"]]

    events: list[tuple[int, str, str]] = []
    for _track_id, g in det.sort_values("frame").groupby("track_id"):
        frames = g["frame"].tolist()
        states = g["zone"].tolist()
        points = list(zip(((g.x1 + g.x2) / 2).tolist(), g.y2.tolist()))
        if len(frames) < 2:
            continue
        events.extend(_confirmed_transitions(frames, states, points, min_dwell_frames))

    flow_rows = []
    for frame, from_zone, to_zone in events:
        sec = int(det.loc[det.frame == frame, "sec"].iloc[0]) if frame in det.frame.values else frame
        if to_zone != OUTSIDE:
            flow_rows.append((sec, to_zone, "inflow"))
        if from_zone != OUTSIDE:
            flow_rows.append((sec, from_zone, "outflow"))

    flows = pd.DataFrame(flow_rows, columns=["sec", "zone", "kind"])
    flow_counts = flows.groupby(["sec", "zone", "kind"]).size().unstack("kind", fill_value=0)
    flow_counts = flow_counts.reindex(columns=["inflow", "outflow"], fill_value=0).reset_index()

    out = counts.merge(flow_counts, on=["sec", "zone"], how="left")
    out[["inflow", "outflow"]] = out[["inflow", "outflow"]].fillna(0).astype(int)
    return out


def load_zones_from_json(path: str) -> list[Zone]:
    """configs/zones/*.json 형식({"zones": [{"name", "polygon"}, ...]})에서 Zone 목록을 읽는다."""
    import json

    with open(path) as f:
        data = json.load(f)
    return [Zone(name=z["name"], polygon=[tuple(p) for p in z["polygon"]]) for z in data["zones"]]
