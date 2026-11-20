"""python -m pytest forecast/serving/test_minute_forecast.py  (pytest 없으면 각 test_ 함수를 직접 호출)"""
import tempfile
from pathlib import Path

import numpy as np, pandas as pd

from forecast.serving import minute_forecast as mf

META = {"a": {"area_m2": 40, "width_m": 4, "exit_width_m": 1.5, "exit_angle_deg": 90, "shape": "bottleneck"}}


def _folder(n_sec=1800, surge_at=1300):
    """30분짜리 가짜 영상 CSV: 평소 유입 0.5명/초, surge_at부터 2명/초로 늘고 출구(1.5m)는 못 따라가 인원이 쌓인다."""
    d = Path(tempfile.mkdtemp())
    sec = np.arange(n_sec)
    inflow = np.where(sec >= surge_at, 2, sec % 2)  # 평소 0/1 번갈아 = 0.5명/초
    outflow = np.where(sec >= surge_at, 1, inflow)
    count = 5 + np.cumsum(inflow - outflow)
    pd.DataFrame({"sec": sec, "zone": "a", "count": count, "inflow": inflow, "outflow": outflow}).to_csv(d / "crowd_observations.csv", index=False)
    pd.DataFrame({"sec": sec, "zone": "a", "angle_deg": 90.0}).to_csv(d / "flow.csv", index=False)
    return d


def test_outputs_and_no_lookahead():
    d = _folder()
    r = mf.run_minute_forecast(d, META)
    fc = r["forecasts"]
    assert list(fc.columns) == mf.FC_COLUMNS and set(fc.horizon_min) == {5, 10, 15}
    assert fc.last_observed_sec.min() == mf.HIST * mf.STEP  # 입력 15분이 찬 뒤부터
    assert ((fc.target_sec - fc.last_observed_sec) == fc.horizon_min * 60).all()
    # 실제 값이 있는 행은 backtest, 영상 끝을 넘으면 forecast
    assert (fc.status.eq("backtest") == fc.actual.notna()).all() and (fc.target_sec[fc.status == "forecast"] > 1800).all()
    assert (Path(r["forecast_path"]).exists() and Path(r["alarm_path"]).exists())


def test_correction_uses_only_past():
    t = np.arange(0, 600, 10.0)
    n0, delta = np.zeros(len(t)), np.full(len(t), 10.0)
    actual = np.zeros(len(t))  # 모델은 계속 +10, 실제는 0 -> 충분히 쌓이면 w=0
    w = mf.correct(t, n0, delta, actual, h=300)
    first = np.argmax(w < 1)
    assert (w[:first] == 1).all() and w[-1] == 0.0
    assert ((t[first] - 300 - t[:first + 1]) >= 0).sum() >= mf.MIN_MATURED  # 바뀐 시점에 지난 예측이 MIN_MATURED개 이상


def test_alarm_when_density_high():
    d = _folder()
    meta = {"a": dict(META["a"], area_m2=2.0)}  # 면적을 아주 작게 -> 밀도가 기준을 넘는다
    r = mf.run_minute_forecast(d, meta)
    assert len(r["alarms"]) > 0 and set(r["alarms"].level) <= {"주의", "위험"}


def test_too_short_video():
    r = mf.run_minute_forecast(_folder(n_sec=600, surge_at=10**6), META)
    assert r["info"]["zones_too_short"] == ["a"] and r["forecasts"].empty


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
