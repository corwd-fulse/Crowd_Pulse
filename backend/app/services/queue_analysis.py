"""대기열(Queue) 점수 산출, 규칙 기반 (bottleneck.py FR-ZON-04의 형제 지표).

bottleneck.py는 "인원 급증 + 이동량 급감 + 유입유출 불균형"의 가중합으로 병목을 본다.
대기열은 그와 다른 패턴이다 — 급증이 없어도, 사람이 많고(density) 거의 안 움직이는(stillness)
상태가 한 구역에서 일정 시간 이상 같이 나타나면 "줄을 서서 기다리는 중"으로 본다. 두 조건이
동시에 성립해야 하므로(사람이 적으면 안 서 있어도 대기열이 아니고, 사람이 많아도 잘 흐르면
대기열이 아니다) 곱으로 합성한다. 병목점수와 같은 구역 시계열(count, motion)을 그대로 쓴다.
"""

from __future__ import annotations

import pandas as pd

WINDOW_S = 10  # bottleneck.py와 같은 윈도우 (스펙 제안값)
MIN_DENSITY_RATIO = 0.3  # 구역 관측 최대 인원 대비 이 비율 미만이면 "줄" 자체가 성립 안 함
MOTION_STILL_RATIO = 0.3  # 구역 관측 최대 motion 대비 이 비율 미만이면 "정체"로 본다


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_queue_scores(
    timeseries: pd.DataFrame,  # columns: sec, zone, count, motion
    window_s: int = WINDOW_S,
) -> pd.DataFrame:
    """반환: columns = [window_start, zone, queue_score, density, stillness, is_queue].

    queue_score = 100 * density * stillness (둘 다 0~1로 정규화). density/stillness는
    그 구역이 관측한 최대 인원/최대 motion 대비 비율 — bottleneck.py와 같은 이유로
    (구역마다 규모가 다름) 절대값 대신 구역 자체 기준으로 정규화한다.
    """
    ts = timeseries.copy()
    ts["window"] = ts["sec"] // window_s * window_s

    rows = []
    for zone, g in ts.groupby("zone"):
        zone_capacity = max(g["count"].max(), 1)
        zone_motion_capacity = max(g["motion"].max(skipna=True) or 0.0, 1e-6)

        agg = g.groupby("window", as_index=False).agg(count=("count", "mean"), motion=("motion", "mean"))
        for _, row in agg.iterrows():
            density = _clip01(row["count"] / zone_capacity)
            stillness = 0.0 if pd.isna(row["motion"]) else _clip01(1 - row["motion"] / zone_motion_capacity)
            is_queue = density >= MIN_DENSITY_RATIO and stillness >= (1 - MOTION_STILL_RATIO)
            rows.append(
                {
                    "window_start": int(row["window"]),
                    "zone": zone,
                    "queue_score": round(100 * density * stillness, 1),
                    "density": round(density, 3),
                    "stillness": round(stillness, 3),
                    "is_queue": is_queue,
                }
            )

    return pd.DataFrame(rows, columns=["window_start", "zone", "queue_score", "density", "stillness", "is_queue"])
