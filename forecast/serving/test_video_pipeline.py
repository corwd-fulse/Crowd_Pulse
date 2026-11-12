"""run_video_forecast 검증. 저장소 루트에서: python -m forecast.serving.test_video_pipeline
샘플: forecast/video_demo_crowd_observations.csv, video_demo_flow.csv (팀 저장소 backend/samples, 60초)."""
import shutil, tempfile
from pathlib import Path
import numpy as np, pandas as pd

from .video_pipeline import COLUMNS, MIN_IF_WINDOWS, run_video_forecast

F = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="vid_test_"))
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(("통과 " if cond else "실패 ") + name + (f" — {detail}" if detail else ""))


def make_dir(name, crowd, flow=None):
    d = TMP / name; d.mkdir()
    crowd.to_csv(d / "crowd_observations.csv", index=False)
    if flow is not None:
        flow.to_csv(d / "flow.csv", index=False)
    return d


crowd0, flow0 = pd.read_csv(F / "video_demo_crowd_observations.csv"), pd.read_csv(F / "video_demo_flow.csv")

# 1) 샘플 60초
res = run_video_forecast(make_dir("sample", crowd0, flow0), horizon_s=10, base_time="2026-10-31 21:00")
fc = res["forecasts"]
check("컬럼 순서", list(fc.columns) == COLUMNS, str(list(fc.columns)))
check("target = last_observed + 10초", ((fc.target_sec - fc.last_observed_sec) == 10).all() and ((fc.target_at - fc.last_observed_at) == pd.Timedelta(seconds=10)).all())
check("구역·모델별 운용 예측 1개씩(split=forecast, actual 없음)", (fc[fc.split == "forecast"].groupby(["zone_id", "model"]).size() == 1).all() and fc[fc.split == "forecast"].actual.isna().all())
cnt = crowd0.set_index(["zone", "sec"])["count"]
pe = fc[(fc.model == "persistence")]
check("persistence = 마지막 관측 시각의 인원", np.allclose(pe.value, [cnt[(z, s)] for z, s in zip(pe.zone_id, pe.last_observed_sec)]))
check("60초 영상에서 LightGBM은 학습 불가 표시", (fc[fc.model == "lightgbm"].status == "insufficient_history").all())
check("이상 이벤트 생성(규칙 + Isolation Forest)", len(res["anomalies"]) > 0 and res["info"]["iforest"] == "ok", f"{len(res['anomalies'])}건")

# 2) 미래 값 누수 없음: 50초 이후 인원을 0으로 바꿔도 last_observed_sec < 40인 예측(대상 50초 미만)은 그대로
c2 = crowd0.copy(); c2.loc[c2.sec >= 50, "count"] = 0
fc2 = run_video_forecast(make_dir("leak", c2, flow0), horizon_s=10, anomaly=False)["forecasts"]
k = ["zone_id", "model", "split", "last_observed_sec"]
m = fc.merge(fc2, on=k, suffixes=("", "_2")); m = m[(m.target_sec < 50) & (m.status == "ok")]
check("미래 값 누수 없음", len(m) > 0 and np.allclose(m.value, m.value_2), f"비교 {len(m)}행")

# 3) 10분짜리 합성 시계열: LightGBM이 그 자리에서 학습되는지 (성능 확인 아님, 동작 확인)
rng = np.random.default_rng(0); t = np.arange(600)
rows, frows = [], []
for z, base in (("entrance", 8), ("inner", 18)):
    cnt_s = base + 4 * np.sin(t / 30) + rng.normal(0, 0.8, len(t)); cnt_s[300:320] += 10
    rows += [{"sec": s, "zone": z, "count": round(v, 1), "inflow": int(rng.poisson(0.3)), "outflow": int(rng.poisson(0.3))} for s, v in zip(t, cnt_s)]
    frows += [{"sec": s, "zone": z, "motion": abs(0.01 + 0.003 * np.cos(s / 20) + rng.normal(0, 0.001))} for s in t]
r3 = run_video_forecast(make_dir("long", pd.DataFrame(rows), pd.DataFrame(frows)), horizon_s=10)
lg = r3["forecasts"][r3["forecasts"].model == "lightgbm"]
check("10분 영상에서 LightGBM 학습·예측(status ok)", (lg.status == "ok").all() and np.isfinite(lg.value).all(), f"백테스트 MAE {r3['info']['backtest_mae']}")

# 4) Isolation Forest 윈도우 경계 29/30 (명세서 TC-ANO-04)
for secs, expect in ((39, False), (40, True)):
    r4 = run_video_forecast(make_dir(f"win{secs}", crowd0[crowd0.sec < secs], flow0[flow0.sec < secs]), horizon_s=1)
    check(f"윈도우 {secs - 10}개 -> Isolation Forest {'학습' if expect else '건너뜀(규칙만)'}", (r4["info"]["iforest"] == "ok") == expect, r4["info"]["iforest"][:40])

# 5) 너무 짧은 영상, 잘못된 CSV
r5 = run_video_forecast(make_dir("short", crowd0[crowd0.sec < 12], flow0[flow0.sec < 12]), horizon_s=10, anomaly=False)["forecasts"]
check("짧은 영상 -> 학습 모델은 insufficient_history", (r5[r5.model.isin(["ridge_ar", "lightgbm"])].status == "insufficient_history").all())
bad = TMP / "pop"; bad.mkdir(); pd.DataFrame({"일자": [20261031], "시간": [0], "행정동코드": ["11110515"], "생활인구합계": [1.0]}).to_csv(bad / "crowd_observations.csv", index=False)
try:
    run_video_forecast(bad); check("생활인구 CSV를 넣으면 오류", False)
except ValueError as e:
    check("생활인구 CSV를 넣으면 오류", True, str(e)[:50])

print("\n판정:", "전부 통과" if ok_all else "실패 있음", f"(임시 폴더 {TMP})")
