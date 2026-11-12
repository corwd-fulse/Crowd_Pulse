import pandas as pd

from app.services.bottleneck import compute_bottleneck_scores


def _timeseries(sec_count_motion_flow):
    rows = [
        {"sec": sec, "zone": "A", "count": count, "motion": motion, "inflow": inflow, "outflow": outflow}
        for sec, count, motion, inflow, outflow in sec_count_motion_flow
    ]
    return pd.DataFrame(rows)


def test_congested_window_scores_higher_than_calm():
    # 혼잡 구간: 인원 급증 + 이동량 급감 + 유입 과다. 비혼잡 구간: 안정적.
    calm = [(s, 10, 5.0, 2, 2) for s in range(0, 10)]
    congested = [(s, 10 + (s - 10) * 3, max(5.0 - (s - 10) * 0.6, 0.1), 6, 0) for s in range(10, 20)]
    ts = _timeseries(calm + congested)

    scores = compute_bottleneck_scores(ts, window_s=10)
    calm_score = scores[scores.window_start == 0]["score"].iloc[0]
    congested_score = scores[scores.window_start == 10]["score"].iloc[0]
    assert congested_score > calm_score


def test_short_video_produces_no_score():
    ts = _timeseries([(0, 10, 5.0, 1, 1)])
    scores = compute_bottleneck_scores(ts, window_s=10)
    assert scores.empty


def test_tiny_motion_dip_does_not_saturate_slowdown():
    # 이 구역의 평소(정상 동작) motion 규모는 1.0. 10~19초엔 이미 0.05로 낮은 상태였다가
    # 20~29초에 0.02로 한 번 더 내려간다. 이전 motion(0.05)으로 나누면 (0.05-0.02)/0.05=0.6로
    # 크게 보이지만, 구역이 평소 낼 수 있는 규모(1.0) 대비로는 0.03에 불과한 무시할 변화다.
    normal = [(s, 10, 1.0, 1, 1) for s in range(0, 10)]
    low = [(s, 10, 0.05, 1, 1) for s in range(10, 20)]
    lower = [(s, 10, 0.02, 1, 1) for s in range(20, 30)]
    ts = _timeseries(normal + low + lower)

    scores = compute_bottleneck_scores(ts, window_s=10)
    dip_slowdown = scores[scores.window_start == 20]["slowdown"].iloc[0]
    assert dip_slowdown < 0.1


def test_tracker_none_drops_imbalance_and_renormalizes():
    rows = [
        {"sec": s, "zone": "A", "count": 10, "motion": 5.0, "inflow": float("nan"), "outflow": float("nan")}
        for s in range(20)
    ]
    ts = pd.DataFrame(rows)
    scores = compute_bottleneck_scores(ts, window_s=10)
    assert not scores.empty
    assert (scores["imbalance"] == 0).all()


def test_risk_level_buckets_match_thresholds():
    calm = [(s, 10, 5.0, 2, 2) for s in range(0, 10)]
    congested = [(s, 10 + (s - 10) * 3, max(5.0 - (s - 10) * 0.6, 0.1), 6, 0) for s in range(10, 20)]
    ts = _timeseries(calm + congested)

    scores = compute_bottleneck_scores(ts, window_s=10)
    assert set(scores["risk_level"]) <= {"낮음", "주의", "높음"}
    assert all(
        (row.risk_level == "높음") == (row.score >= 70)
        or (row.risk_level == "주의") == (40 <= row.score < 70)
        or (row.risk_level == "낮음") == (row.score < 40)
        for row in scores.itertuples()
    )
