"""병목 점수 산출, 규칙 기반 (FR-ZON-04).

구역 시계열(count, inflow, outflow, motion)을 10초 윈도우로 묶어 인원
증가율·이동량 감소·유입유출 불균형을 가중합한 0~100 연구용 지표를 만든다.
실제 사고/위험 확정 판정이 아니다.
"""

from __future__ import annotations

import pandas as pd

WINDOW_S = 10  # 스펙 제안값
WEIGHTS = (0.40, 0.35, 0.25)  # growth, slowdown, imbalance (스펙 제안값)
RISK_THRESHOLDS = (40, 70)  # FR-ZON-05 제안값: <40 낮음, 40~70 주의, >=70 높음


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_bottleneck_scores(
    timeseries: pd.DataFrame,  # columns: sec, zone, count, inflow, outflow, motion
    window_s: int = WINDOW_S,
    weights: tuple[float, float, float] = WEIGHTS,
) -> pd.DataFrame:
    """반환: columns = [window_start, zone, score, growth, slowdown, imbalance]."""
    ts = timeseries.copy()
    ts["window"] = ts["sec"] // window_s * window_s

    has_flow = ts["inflow"].notna().any() if "inflow" in ts else False
    w1, w2, w3 = weights
    if not has_flow:
        # tracker='none': 불균형 항목 제외하고 나머지 가중치 재정규화
        total = w1 + w2
        w1, w2, w3 = w1 / total, w2 / total, 0.0

    rows = []
    for zone, g in ts.groupby("zone"):
        # 구역 자체 관측 최대 인원을 "용량" 기준으로 써서 growth를 정규화한다.
        # 이전 윈도우 값으로 나누면(예: 4명 -> 9명) 사람이 원래 적은 구역에서
        # 작은 절대 변화도 비율상 쉽게 1.0으로 포화돼 과대평가되는 문제가 있었다.
        zone_capacity = max(g["count"].max(), 1)
        # slowdown도 같은 문제가 있었다: 이전 motion으로 나누면 motion이 거의 0에 가까운
        # 순간 바로 다음 작은 절대 변화도 비율상 1.0에 가깝게 튄다. 구역의 관측 최대
        # motion을 기준치로 써서 절대 변화량 기준으로 정규화한다.
        zone_motion_capacity = max(g["motion"].max(skipna=True) or 0.0, 1e-6)

        agg = {"count": "mean", "motion": "mean"}
        if has_flow:
            agg.update(inflow="sum", outflow="sum")
        g = g.groupby("window", as_index=False).agg(agg).sort_values("window").reset_index(drop=True)

        if len(g) < 2:
            continue  # 윈도우보다 짧은 영상: 점수 산출 안 함

        prev_count = prev_motion = None
        for _, row in g.iterrows():
            growth = 0.0 if prev_count is None else _clip01((row["count"] - prev_count) / zone_capacity)

            if prev_motion is None or pd.isna(prev_motion) or pd.isna(row.get("motion")):
                slowdown = 0.0
            else:
                slowdown = _clip01((prev_motion - row["motion"]) / zone_motion_capacity)

            if has_flow:
                total_flow = row["inflow"] + row["outflow"]
                imbalance = _clip01(abs(row["inflow"] - row["outflow"]) / max(total_flow, 1))
            else:
                imbalance = 0.0

            score = round(100 * (w1 * growth + w2 * slowdown + w3 * imbalance), 1)
            rows.append(
                {
                    "window_start": int(row["window"]),
                    "zone": zone,
                    "score": score,
                    "growth": round(growth, 3),
                    "slowdown": round(slowdown, 3),
                    "imbalance": round(imbalance, 3),
                }
            )
            prev_count, prev_motion = row["count"], row.get("motion")

    out = pd.DataFrame(rows, columns=["window_start", "zone", "score", "growth", "slowdown", "imbalance"])
    out["risk_level"] = out["score"].apply(_risk_level)
    return out


def _risk_level(score: float, thresholds: tuple[int, int] = RISK_THRESHOLDS) -> str:
    """FR-ZON-05: 점수를 낮음/주의/높음 3단계로 분류한다 (연구용 지표, 실제 위험 확정 아님)."""
    low, high = thresholds
    if score >= high:
        return "높음"
    if score >= low:
        return "주의"
    return "낮음"
