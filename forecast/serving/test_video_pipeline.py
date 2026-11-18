"""run_video_forecast 검증. 저장소 루트에서: python -m forecast.serving.test_video_pipeline
샘플: forecast/video_demo_crowd_observations.csv, video_demo_flow.csv (팀 저장소 backend/samples, 60초)."""
import json, shutil, tempfile
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
res = run_video_forecast(make_dir("sample", crowd0, flow0), horizon_s=10, base_time="2026-10-31 21:00", target="raw")
fc = res["forecasts"]
check("컬럼 순서", list(fc.columns) == COLUMNS, str(list(fc.columns)))
check("target = last_observed + 10초", ((fc.target_sec - fc.last_observed_sec) == 10).all() and ((fc.target_at - fc.last_observed_at) == pd.Timedelta(seconds=10)).all())
check("구역·모델별 운용 예측 1개씩(split=forecast, actual 없음)", (fc[fc.split == "forecast"].groupby(["zone_id", "model"]).size() == 1).all() and fc[fc.split == "forecast"].actual.isna().all())
cnt = crowd0.set_index(["zone", "sec"])["count"]
pe = fc[(fc.model == "persistence")]
check("영상 초반부터 매초 예측(직전값 기준선은 5초부터)", fc[(fc.model == "persistence") & (fc.split == "backtest")].last_observed_sec.min() <= 5)
check("persistence = 마지막 관측 시각의 인원", np.allclose(pe.value, [cnt[(z, s)] for z, s in zip(pe.zone_id, pe.last_observed_sec)]))
check("60초 영상에서 LightGBM은 학습 불가 표시", (fc[fc.model == "lightgbm"].status == "insufficient_history").all())
check("이상 이벤트 생성(규칙 + Isolation Forest)", len(res["anomalies"]) > 0 and res["info"]["iforest"].startswith("ok"), f"{len(res['anomalies'])}건")

# 2) 미래 값 누수 없음: 50초 이후 인원을 0으로 바꿔도 last_observed_sec < 40인 예측(대상 50초 미만)은 그대로
c2 = crowd0.copy(); c2.loc[c2.sec >= 50, "count"] = 0
fc2 = run_video_forecast(make_dir("leak", c2, flow0), horizon_s=10, anomaly=False, target="raw")["forecasts"]
fm1 = run_video_forecast(make_dir("leak_m1", crowd0, flow0), horizon_s=10, anomaly=False)["forecasts"]
fm2 = run_video_forecast(make_dir("leak_m2", c2, flow0), horizon_s=10, anomaly=False)["forecasts"]
k = ["zone_id", "model", "split", "last_observed_sec"]
m = fc.merge(fc2, on=k, suffixes=("", "_2")); m = m[(m.target_sec < 50) & (m.status == "ok")]
check("미래 값 누수 없음(원시 인원)", len(m) > 0 and np.allclose(m.value, m.value_2), f"비교 {len(m)}행")
mm = fm1.merge(fm2, on=k, suffixes=("", "_2")); mm = mm[(mm.target_sec < 50) & (mm.status == "ok")]
check("미래 값 누수 없음(10초 평균)", len(mm) > 0 and np.allclose(mm.value, mm.value_2), f"비교 {len(mm)}행")

# 3) 10분짜리 합성 시계열: LightGBM이 그 자리에서 학습되는지 (성능 확인 아님, 동작 확인)
rng = np.random.default_rng(0); t = np.arange(600)
rows, frows = [], []
for z, base in (("entrance", 8), ("inner", 18)):
    cnt_s = base + 4 * np.sin(t / 30) + rng.normal(0, 0.8, len(t)); cnt_s[300:320] += 10
    rows += [{"sec": s, "zone": z, "count": round(v, 1), "inflow": int(rng.poisson(0.3)), "outflow": int(rng.poisson(0.3))} for s, v in zip(t, cnt_s)]
    frows += [{"sec": s, "zone": z, "motion": abs(0.01 + 0.003 * np.cos(s / 20) + rng.normal(0, 0.001))} for s in t]
r3 = run_video_forecast(make_dir("long", pd.DataFrame(rows), pd.DataFrame(frows)), horizon_s=10)
lg = r3["forecasts"][r3["forecasts"].model == "lightgbm"]
okl = lg[lg.status == "ok"]
check("10분 영상에서 LightGBM 학습·예측(학습 행 200개가 쌓인 뒤 ok)", len(okl) > 300 and np.isfinite(okl.value).all() and (lg[lg.last_observed_sec < 200].status == "insufficient_history").all(), f"ok {len(okl)}행")

# 4) Isolation Forest 윈도우 경계 29/30 (명세서 TC-ANO-04)
for secs, expect in ((39, False), (40, True)):
    r4 = run_video_forecast(make_dir(f"win{secs}", crowd0[crowd0.sec < secs], flow0[flow0.sec < secs]), horizon_s=1)
    check(f"윈도우 {secs - 10}개 -> Isolation Forest {'학습' if expect else '건너뜀(규칙만)'}", r4["info"]["iforest"].startswith("ok") == expect, r4["info"]["iforest"][:40])

# 5) 너무 짧은 영상, 잘못된 CSV
r5 = run_video_forecast(make_dir("short", crowd0[crowd0.sec < 12], flow0[flow0.sec < 12]), horizon_s=10, anomaly=False)["forecasts"]
check("짧은 영상 -> 학습 모델은 insufficient_history", (r5[r5.model.isin(["ridge_ar", "lightgbm"])].status == "insufficient_history").all())
bad = TMP / "pop"; bad.mkdir(); pd.DataFrame({"일자": [20261031], "시간": [0], "행정동코드": ["11110515"], "생활인구합계": [1.0]}).to_csv(bad / "crowd_observations.csv", index=False)
try:
    run_video_forecast(bad); check("생활인구 CSV를 넣으면 오류", False)
except ValueError as e:
    check("생활인구 CSV를 넣으면 오류", True, str(e)[:50])

# 6) 10초 평균 대상(mean10, 기본): persistence = 지금까지 10초 평균, 여러 예측 시간 출력
r6 = run_video_forecast(make_dir("mean10", crowd0, flow0), horizons=(10, 30), anomaly=False)["forecasts"]
p6 = r6[(r6.model == "persistence") & (r6.status == "ok") & (r6.horizon_s == 10)]
ref = {(z, s): v for z, g in crowd0.sort_values("sec").groupby("zone") for s, v in zip(g.sec, g["count"].rolling(10).mean())}
check("mean10: persistence = 최근 10초 평균, 예측 시간 10·30초 모두 출력",
      np.allclose(p6.value, [ref[(z, s)] for z, s in zip(p6.zone_id, p6.last_observed_sec)], atol=0.01) and set(r6.horizon_s) == {10, 30})

# 7) 3단계 판정 (anomaly_levels)
from .. import anomaly_levels as al
from .. import anomaly_video as av
check("흔들림 방지: 2초짜리 신호는 단계를 올리지 않음", al.hysteresis([0, 1, 1, 0, 0, 0]).max() == 0)
check("3초 유지 시 주의, 5초 풀리면 내림", list(al.hysteresis([1, 1, 1, 0, 0, 0, 0, 0])) == [0, 0, 1, 1, 1, 1, 1, 0])
check("주의가 20초 연속이면 높음", al.hysteresis([1] * 25)[-1] == 2 and al.hysteresis([1] * 21)[-1] == 1)
flat = pd.DataFrame({"sec": range(60), "zone": "z", "count": 10.0, "motion": 0.01})
for bnv, exp in ((39, 0), (40, 1), (69, 1), (70, 2)):
    bnd = pd.DataFrame({"window_start": range(0, 60, 10), "zone": "z", "score": float(bnv)})
    lv = al.grade_levels(flat, bottleneck=bnd).level_num.iloc[10:19]  # 20초 연속 규칙이 끼기 전 구간
    check(f"병목 {bnv} -> {al.LEVELS[exp]}", lv.max() == exp and lv.min() == exp)
ts0 = av.load_zone_series(F / "video_demo_crowd_observations.csv", F / "video_demo_flow.csv")
inj = av.inject_surge(ts0, "entrance", 40, factor=3.0, motion_factor=0.2)
tl = al.grade_levels(inj, av.rule_events(inj), av.iforest_scores(inj)[0])
e = tl[(tl.zone_id == "entrance") & tl.sec.between(40, 48)]
check("급증 주입 구간이 주의 이상으로 오름", e.level_num.max() >= 1, f"주입 구간 최고 {al.LEVELS[int(e.level_num.max())]}")
ev = al.level_events(tl)
check("단계 이벤트에 근거 문장·지속 시간·최고 시각", len(ev) > 0 and ev.reason.str.len().gt(0).all() and (ev.duration_s >= 1).all() and ev.peak_sec.between(ev.event_sec, ev.end_sec).all(), ev.reason.iloc[0])
txt = " ".join(ev.reason) + " ".join(al.LEVELS)
check("금지 표현 없음(사고 확정, 위험 판정, 위험)", not any(w in txt for w in ("사고 확정", "위험 판정", "위험")))
# 도착 예정 특징: 위 구역(y<600)에서 아래로 50px/s 걷는 사람 1명(10fps, 키 100px). 2초 끝 y=445 -> 600까지 약 3.1초
tt = np.arange(0, 3, 0.1); yy = 300 + 50 * tt
dpath = TMP / "arrival_det.csv"
pd.DataFrame({"frame": range(len(tt)), "t_ms": (tt * 1000).round().astype(int), "track_id": 1, "x1": 450, "y1": yy - 100, "x2": 550, "y2": yy, "conf": 0.9}).to_csv(dpath, index=False)
zz = [{"name": "top", "polygon": [[0, 0], [1000, 0], [1000, 600], [0, 600]]}, {"name": "bottom", "polygon": [[0, 600], [1000, 600], [1000, 1000], [0, 1000]]}]
af = al.arrival_features(dpath, zz).set_index(["zone", "sec"])
check("도착 예정: 2초 시점 아래 구역 arr10=1·위 구역 dep10=1, 0초 시점(이력 1초 미만)은 0",
      af.loc[("bottom", 2), "arr10"] == 1 and af.loc[("top", 2), "dep10"] == 1 and af.loc[("bottom", 0), "arr60"] == 0 and af.loc[("top", 2), "arr60"] == 0)
# 끊긴 추적 잇기: 같은 사람이 1.5초에 ID 1 -> 1.8초부터 ID 2(공백 0.3초, 같은 속도). 멀리 떨어져 생긴 ID 3은 잇지 않는다
d1 = pd.DataFrame({"frame": range(len(tt)), "t_ms": (tt * 1000).round().astype(int), "track_id": np.where(tt < 1.5, 1, 2), "x1": 450, "y1": yy - 100, "x2": 550, "y2": yy, "conf": 0.9})
d1 = d1[(tt < 1.5) | (tt >= 1.8)]
d1 = pd.concat([d1, pd.DataFrame({"frame": [25], "t_ms": [2500], "track_id": [3], "x1": [50], "y1": [100], "x2": [100], "y2": [200], "conf": [0.9]})])
st_ = al.stitch_tracks(d1)
check("끊긴 추적 잇기: 0.3초 공백 같은 경로는 한 ID, 먼 곳의 새 ID는 그대로", st_[st_.raw_track_id == 2].track_id.eq(1).all() and st_[st_.raw_track_id == 3].track_id.eq(3).all())
af2 = al.arrival_features(d1, zz, stitch_gap_s=1.0).set_index(["zone", "sec"])
check("이은 뒤 새로 나타난 사람: 0초에 위 구역 안 1명, 2초에 1명 더(ID 3) -> 위 구역 안 최근 60초 2명, 아래 구역 밖 2명",
      af2.loc[("top", 0), "births_in60"] == 1 and af2.loc[("top", 2), "births_in60"] == 2 and af2.loc[("bottom", 2), "births_out60"] == 2)
full = run_video_forecast(make_dir("full", crowd0, flow0), horizons=(10,))
check("anomaly_timeline.csv 생성(구역별 매초 단계)", full["timeline_path"] is not None and Path(full["timeline_path"]).exists() and len(full["timeline"]) == len(ts0))

# 8) 미리 학습(train_video) -> 서비스(lightgbm_pre, lstm_pre): 규모가 다른 카메라 영상으로 학습, 처음 보는 영상에서 첫 초부터 예측
from ..training import train_video as tv
from .video_pipeline import pre_design, pre_models, _zone_features, load_video_series


def synth(name, scale, phase, seed, n=600):
    r = np.random.default_rng(seed); rows = []
    for z, k in (("a", 1.0), ("b", 1.6)):
        rows += [{"sec": s, "zone": z, "count": max(0.0, scale * k * (1 + 0.5 * np.sin(2 * np.pi * s / 90 + phase)) + r.normal(0, 0.05 * scale)),
                  "inflow": r.poisson(1), "outflow": r.poisson(1)} for s in range(n)]
    return make_dir(name, pd.DataFrame(rows))


PRE, HS = TMP / "pre", (10, 30)
train_dirs = [synth(f"tr{i}", sc, ph, i) for i, (sc, ph) in enumerate(((4, 0.0), (12, 1.0), (30, 2.0)))]
try:
    tv.train(train_dirs[:1], horizons=HS, out_dir=PRE); check("영상 1개면 학습 거부(영상 단위 검증 불가)", False)
except ValueError as e:
    check("영상 1개면 학습 거부(영상 단위 검증 불가)", True, str(e)[:40])
rep = tv.train([TMP], horizons=HS, out_dir=PRE)  # 상위 폴더: 앞 테스트 폴더 포함 전부 찾음, 생활인구 CSV 폴더는 건너뜀
check("형식이 틀린 폴더는 건너뛰고 기록", any("pop" in k for k in rep["skipped_folders"]), str(list(rep["skipped_folders"]))[:60])
cp = rep["compare"]
check("비교표: Naive·LightGBM·LSTM × 예측 시간별 MAE·RMSE", set(cp.model) == {"naive", "lightgbm_pre", "lstm_pre"} and set(cp.horizon_s) == set(HS)
      and cp[["MAE", "RMSE"]].notna().all().all() and (PRE / "compare_video.csv").exists(), "\n" + cp.to_string(index=False))
check("직전값보다 나은 모델 저장 + SHAP 그림", all(v == "교체" for v in rep["saved"].values()) and (PRE / "shap_importance.png").exists(), str(rep["saved"]))
new = synth("unseen", 50, 0.5, 9)
r8 = run_video_forecast(new, horizons=HS, anomaly=False, pre_dir=PRE)
f8, bk8 = r8["forecasts"], r8["info"]["backtest"]
for m in ("lightgbm_pre", "lstm_pre"):
    p = f8[f8.model == m]
    check(f"{m}: 영상 초반(15초 안)부터 예측", p[p.status == "ok"].last_observed_sec.min() <= 15)
    check(f"{m}: 처음 보는 영상에서 직전값보다 나음(10초 뒤)", bk8[f"10s/{m}"]["skill_vs_persistence"] > 0, str(bk8[f"10s/{m}"]))
check("서비스 평가에 RMSE", bk8["10s/lightgbm_pre"]["RMSE"] is not None)
ts8 = load_video_series(new)[0]; g8 = ts8[ts8.zone == "a"]; d8 = pre_design(g8, _zone_features(ts8)["a"], "mean10")
b8 = pre_models("mean10", PRE)["lgbm"][10]
check("SHAP: 기여 합 = 예측값(TreeSHAP 성질)", np.allclose(b8.predict(d8["X"], pred_contrib=True).sum(1), b8.predict(d8["X"]), atol=1e-6))
s8 = f8[(f8.model == "lightgbm_pre") & (f8.status == "ok")].shap_top
check("결과 CSV에 shap_top(상위 기여 특징 3개)", s8.str.count("명").eq(3).all() and (f8[f8.model != "lightgbm_pre"].shap_top == "").all(), s8.iloc[0])
c8 = pd.read_csv(new / "crowd_observations.csv"); c8.loc[c8.sec >= 300, "count"] = 0
f8b = run_video_forecast(make_dir("unseen_leak", c8), horizons=HS, anomaly=False, pre_dir=PRE)["forecasts"]
for m in ("lightgbm_pre", "lstm_pre"):
    mm8 = f8[f8.model == m].merge(f8b[f8b.model == m], on=k + ["horizon_s"], suffixes=("", "_2"))
    mm8 = mm8[(mm8.target_sec < 300) & (mm8.status == "ok")]
    check(f"{m}: 미래 값 누수 없음", len(mm8) > 0 and np.allclose(mm8.value, mm8.value_2, atol=1e-4), f"비교 {len(mm8)}행")
f45 = run_video_forecast(new, horizons=(45,), anomaly=False, pre_dir=PRE)["forecasts"]
check("학습 안 한 예측 시간·모델 파일 없음 -> no_pretrained_model", (f45[f45.model.isin(["lightgbm_pre", "lstm_pre"])].status == "no_pretrained_model").all()
      and (run_video_forecast(new, horizons=(10,), anomaly=False, pre_dir=TMP / "empty")["forecasts"].query("model == 'lstm_pre'").status == "no_pretrained_model").all())
tv.train(train_dirs, horizons=HS, out_dir=PRE, force=True)
man = json.loads((PRE / "manifest.json").read_text(encoding="utf-8"))["models"]
check("재학습하면 이전 모델을 history에 보관", all(man[kk]["previous_version"] and (PRE / "history" / man[kk]["previous_version"] / man[kk]["file"]).exists()
                                        for kk in ("lgbm_mean10_10s", "lstm_mean10")))

print("\n판정:", "전부 통과" if ok_all else "실패 있음", f"(임시 폴더 {TMP})")
