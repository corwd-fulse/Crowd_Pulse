"""군중 이동 속도, m/s 환산 (카메라 보정값이 있을 때만).

optical_flow.py/individual_flow.py는 픽셀 단위까지만 내고 "m/s로는 절대 환산하지 않는다"고
주석에 명시한다 — 영상마다 카메라 거리/각도가 달라 같은 픽셀 이동이 다른 실제 속도를 의미하고,
그 비율(pixels_per_meter)을 자동으로 알 수 없기 때문이다. 이 모듈은 그 보정값을 입력으로
받을 때만 m/s로 바꾼다. 바닥이 화면 안에서 대략 평면이라는 가정의 근사치이며, 원근 때문에
같은 구역 안에서도 멀고 가까운 쪽의 실제 비율이 다를 수 있다(보정 안 함, 연구용 참고 지표).
보정값이 없으면 이 모듈을 호출하지 않으면 된다 — 기존 flow.csv/individual_directions.csv
스키마는 그대로 두고 완전히 새 파일로만 결과를 낸다.
"""

from __future__ import annotations

import json

import pandas as pd

PixelsPerMeter = float | dict[str, float]


def _ratio_for_zone(pixels_per_meter: PixelsPerMeter, zone: str) -> float | None:
    if isinstance(pixels_per_meter, dict):
        return pixels_per_meter.get(zone)
    return pixels_per_meter


def load_calibration_from_json(path: str) -> dict[str, float]:
    """configs/zones/*_calibration.json 형식({"zones": [{"name", "pixels_per_meter"}, ...]})."""
    with open(path) as f:
        data = json.load(f)
    return {z["name"]: float(z["pixels_per_meter"]) for z in data["zones"]}


def compute_zone_speed(motion: pd.DataFrame, fps: float, pixels_per_meter: PixelsPerMeter) -> pd.DataFrame:
    """optical_flow.compute_zone_motion() 출력(magnitude_px는 px/frame)을 m/s로 바꾼다.

    반환: columns = [sec, zone, speed_mps, low_confidence]. 보정값이 없는 구역이거나
    magnitude_px 자체가 없는(low_confidence) 행은 speed_mps=None, low_confidence=True.
    """
    rows = []
    for row in motion.itertuples():
        ratio = _ratio_for_zone(pixels_per_meter, row.zone)
        if ratio is None or ratio <= 0 or pd.isna(row.magnitude_px):
            rows.append({"sec": row.sec, "zone": row.zone, "speed_mps": None, "low_confidence": True})
            continue
        speed_mps = row.magnitude_px * fps / ratio
        rows.append(
            {
                "sec": row.sec,
                "zone": row.zone,
                "speed_mps": round(float(speed_mps), 3),
                "low_confidence": bool(row.low_confidence),
            }
        )
    return pd.DataFrame(rows, columns=["sec", "zone", "speed_mps", "low_confidence"])


def compute_individual_speed(
    individual: pd.DataFrame,
    fps: float,
    pixels_per_meter: float,
    window_frames: int,
) -> pd.DataFrame:
    """individual_flow.compute_individual_directions() 출력(speed_px는 window_frames 프레임
    동안의 누적 이동 거리)을 m/s로 바꾼다. 구역마다 다른 비율이 아니라 영상 전체에 단일
    비율을 쓴다 — 트랙 위치마다 구역을 다시 조회해야 해서 역할을 compute_zone_speed와 분리했다.

    주의: 영상 시작 직후(frame < window_frames)는 실제 비교 구간이 window_frames보다 짧은데도
    같은 window_frames로 나눠 속도를 과소평가한다(individual_flow.py의 같은 창 처리 방식을
    그대로 따른 결과, shortcut).
    """
    if pixels_per_meter <= 0:
        raise ValueError("pixels_per_meter must be positive")
    out = individual[["frame", "t_ms", "track_id", "reliable"]].copy()
    elapsed_s = window_frames / fps
    reliable = individual["reliable"] & individual["speed_px"].notna()
    out["speed_mps"] = None
    out.loc[reliable, "speed_mps"] = (individual.loc[reliable, "speed_px"] / pixels_per_meter / elapsed_s).round(3)
    return out[["frame", "t_ms", "track_id", "speed_mps", "reliable"]]
