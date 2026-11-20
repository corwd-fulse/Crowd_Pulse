"""python -m pytest forecast/bottleneck/test_bottleneck.py"""
import numpy as np, pandas as pd

from forecast.bottleneck.crowdqueue import bottleneck_label, queue_forecast, zone_seconds


def test_label_needs_20_seconds():
    d = np.r_[np.full(5, 1.0), np.full(20, 3.0), np.full(3, 1.0)]
    lab = bottleneck_label(d)
    assert lab[5] and not lab[6] and lab.sum() == 1  # 20초 연속은 시작 시점 하나만 '이후 20초 지속'
    assert not bottleneck_label(np.full(19, 9.0)).any()


def test_zone_seconds_counts_and_passing():
    # 1번: 문 앞(y=1) → 문 통과(y=-0.5), 2번: 뒤쪽(y=3)에 머묾. 통로 폭 1m → 문 앞 면적 2㎡
    fr = [0, 25, 0, 25]
    d = pd.DataFrame({"id": [1, 1, 2, 2], "fr": fr, "x": 0.0, "y": [1.0, -0.5, 3.0, 3.0], "z": 1.7})
    z = zone_seconds(d, 1.0)
    assert z.n_front.tolist() == [1, 0] and z.n_back.tolist() == [1, 1]
    assert z.dens.tolist() == [0.5, 0.0] and z.outflow.tolist() == [0, 1]


def test_queue_drains_at_rate():
    z = pd.DataFrame({"n_front": [10, 10], "n_back": [10, 9], "outflow": [1, 1], "area": 2.0})
    q = queue_forecast(z, 5)
    assert q.tolist() == [5.0, 5.0]  # 남은 20/19명 − 5명 → 문 앞 최대 10명 유지 → 10/2㎡


def test_schedule_features_match_numeric_integral():
    import json
    import numpy as np
    from forecast.bottleneck.sim_models import H, noisy_plan, schedule_features
    row = {"sid": 3, "base": 0.5, "bumps": json.dumps([[1200.0, 200.0, 2.0], [2500.0, 300.0, 1.0]])}
    t = np.array([0.0, 1000.0, 2400.0, 3000.0])
    f = schedule_features(row, t, cap=2.6)
    base, bumps = noisy_plan(row)
    lam = lambda s: base + sum(h * np.exp(-0.5 * ((s - c) / w) ** 2) for c, w, h in bumps)
    for h in H:  # 1초 합과 적분이 거의 같아야 함
        num = [lam(np.arange(x, x + h) + 0.5).sum() for x in t]
        assert np.allclose(f[f"plan_in_{h // 60}m"], num, rtol=1e-3)
    assert all(abs(c - o) <= 180 for (c, _, _), (o, _, _) in zip(bumps, json.loads(row["bumps"])))  # 시각 오차 ±3분
    assert f["plan_next_in"][-1] == 3600.0 and f["plan_next_in"][0] == bumps[0][0]  # 남은 몰림 없음 / 첫 몰림까지


def test_hide_and_events():
    import json
    import numpy as np
    from forecast.bottleneck.sim_models import noisy_plan
    from forecast.bottleneck.sim_announce import events
    row = {"sid": 5, "base": 0.5, "bumps": json.dumps([[1200.0, 200.0, 2.0], [2500.0, 300.0, 1.0]])}
    assert noisy_plan(row, hide=1.0)[1] == [] and noisy_plan(row, hide=0.0)[1] == noisy_plan(row)[1]
    d = np.array([0, 3, 0, 3, 3, 3, 0, 3, 3])  # 1칸짜리는 사건 아님, 2칸 이상만
    assert list(events(d, np.arange(len(d)) * 10 + 10, 3.0)) == [40, 80]


def test_video_bins_from_csv(tmp_path=None):
    import tempfile
    from pathlib import Path
    from forecast.bottleneck.sim_serving import video_bins
    d = Path(tmp_path or tempfile.mkdtemp())
    sec = np.arange(30)
    pd.DataFrame({"sec": sec, "zone": "a", "count": sec % 10, "inflow": 1, "outflow": 0}).to_csv(d / "crowd_observations.csv", index=False)
    pd.DataFrame({"sec": sec, "zone": "a", "angle_deg": 90.0}).to_csv(d / "flow.csv", index=False)
    meta = {"a": {"area_m2": 50, "width_m": 5, "exit_width_m": 2, "exit_angle_deg": 90, "shape": "bottleneck"}}
    g = video_bins(d, meta)
    assert list(g.n) == [9, 9, 9] and list(g.inflow) == [10, 10, 10]  # 10초 칸: 인원 = 칸 끝 값, 유입 = 합
    assert np.allclose(g.heading, 1.0) and np.allclose(g.density, 9 / 50) and g.speed.isna().all()
