"""영상 분석 CSV -> 전처리 -> 구역 인원 단기 예측 + 이상 흐름 탐지 -> 결과 CSV.
입력: 팀 영상 파이프라인(backend/app/services/pipeline.py run_zone_pipeline)이 만든 폴더 또는 crowd_observations.csv 경로.
      crowd_observations.csv(sec, zone, count, inflow, outflow) 필수, 같은 폴더의 flow.csv(sec, zone, motion)는 있으면 사용.

Python:
    from forecast.serving import run_video_forecast
    res = run_video_forecast("out/분석결과폴더", horizon_s=10, base_time="2026-10-31 21:00")
    res["forecast_path"], res["anomaly_path"], res["forecasts"], res["anomalies"], res["info"]
CLI (저장소 루트에서): python -m forecast.serving.video_pipeline 폴더 [--horizon 10] [--base-time "2026-10-31 21:00"]

예측 방식 (생활인구 모델과 달리 미리 학습된 모델이 없다 -> 학습이 필요한 모델은 들어온 영상의 앞부분으로 그 자리에서 학습):
  persistence    : 직전 값 그대로 (학습 없음, 기준선)
  moving_avg_5s  : 최근 5초 평균 (학습 없음)
  ridge_ar       : 최근 5초 변화로 H초 뒤 변화 예측 (영상 앞 60%로 학습, 학습 행 20개 이상)
  lightgbm       : 같은 특징 + 유입·유출·이동량 (영상 앞 60%로 학습, 학습 행 200개 이상 = 대략 4분 이상 영상)
평가: 영상 뒤 40%에서 매초 'H초 뒤'를 예측해 실제와 비교(split=backtest), 마지막 시각 + H초 예측 1개(split=forecast, actual 없음).
연구용 참고 지표이며 현장 상황 판단은 운영자 확인이 필요하다."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

from .. import anomaly_video as av

P = 5  # 입력으로 쓰는 과거 초 수
TRAIN_FRAC = 0.6
MIN_TRAIN = {"ridge_ar": 20, "lightgbm": 200}
MIN_IF_WINDOWS = 30  # 명세서 TC-ANO-04: 윈도우 29개는 Isolation Forest 학습 불가, 30개부터 가능
MODELS = ("persistence", "moving_avg_5s", "ridge_ar", "lightgbm")
COLUMNS = ["zone_id", "model", "split", "last_observed_sec", "target_sec", "last_observed_at", "target_at", "value", "actual", "status"]


def load_video_series(source):
    """폴더 또는 crowd_observations.csv -> 구역별 1초 간격 시계열[sec, zone, count, inflow, outflow, motion]."""
    p = Path(source)
    crowd = p / "crowd_observations.csv" if p.is_dir() else p
    if not crowd.exists():
        raise FileNotFoundError(f"crowd_observations.csv가 없습니다: {crowd}")
    c = pd.read_csv(crowd, encoding="utf-8-sig")
    need = {"sec", "zone", "count"}
    if not need <= set(c.columns):
        raise ValueError(f"{crowd.name}: 영상 구역 CSV 형식이 아닙니다(필요 컬럼 {sorted(need)}, 받은 컬럼 {list(c.columns)[:6]})")
    for col in ("inflow", "outflow"):
        if col not in c:
            c[col] = np.nan  # tracker 없이 만든 결과(유입/유출 없음)
    flow = crowd.parent / "flow.csv"
    if flow.exists():
        f = pd.read_csv(flow, encoding="utf-8-sig")[["sec", "zone", "motion"]]
        c = c.merge(f, on=["sec", "zone"], how="left")
    else:
        c["motion"] = np.nan
    out = []
    for z, g in c.groupby("zone"):
        g = g.sort_values("sec").drop_duplicates("sec", keep="last").set_index("sec")
        g = g.reindex(range(int(g.index.min()), int(g.index.max()) + 1))  # 빠진 초를 채워 1초 간격으로
        g["count"], g["motion"] = g["count"].interpolate(limit=2, limit_area="inside"), g["motion"].interpolate(limit=2, limit_area="inside")
        g[["inflow", "outflow"]] = g[["inflow", "outflow"]].fillna(0) if c.inflow.notna().any() else g[["inflow", "outflow"]]
        g["zone"] = z
        out.append(g.rename_axis("sec").reset_index())
    ts = pd.concat(out, ignore_index=True)[["sec", "zone", "count", "inflow", "outflow", "motion"]]
    info = {"crowd_csv": str(crowd), "flow_csv": str(flow) if flow.exists() else None, "zones": sorted(ts.zone.unique().tolist()),
            "seconds": int(ts.sec.max() - ts.sec.min() + 1), "missing_count_cells": int(ts["count"].isna().sum())}
    return ts, info


def _features(g, t):
    """시각 t까지의 값만으로 만든 특징(미래 값 미사용)."""
    c = g["count"].to_numpy(float)
    X = np.column_stack([c[t]] + [c[t - k] - c[t] for k in range(1, P)])
    extra = [g[col].rolling(P, min_periods=1).sum().to_numpy(float)[t] for col in ("inflow", "outflow")] + [g["motion"].rolling(P, min_periods=1).mean().to_numpy(float)[t]]
    return X, np.column_stack([X] + extra)


def _ridge(X, y, alpha=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = (X - mu) / sd
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (y - y.mean()))
    return lambda Z: ((Z - mu) / sd) @ w + y.mean()


def forecast_zone(g, H, models=MODELS, train_frac=TRAIN_FRAC):
    """한 구역: 뒤 40% 백테스트 + 마지막 시각 + H초 예측. 반환: 행 목록(dict)."""
    g = g.reset_index(drop=True); c = g["count"].to_numpy(float); n = len(c); sec = g.sec.to_numpy()
    split = int(n * train_frac)
    tr = np.arange(P - 1, max(P - 1, split - H))              # 정답(t+H)이 학습 구간 안
    te = np.arange(max(split, P - 1), n - H)                    # 백테스트: t에서 t+H 예측
    last = np.array([n - 1])                                    # 운용: 마지막 시각에서 +H
    ok_hist = lambda t: ~np.isnan(np.column_stack([c[t - k] for k in range(P)])).any(1)
    tr = tr[ok_hist(tr) & ~np.isnan(c[tr + H])] if len(tr) else tr
    fitted = {}
    for m in ("ridge_ar", "lightgbm"):
        if m not in models:
            continue
        if len(tr) < MIN_TRAIN[m]:
            fitted[m] = None; continue
        Xb, Xf = _features(g, tr); y = c[tr + H] - c[tr]
        if m == "ridge_ar":
            fitted[m] = (_ridge(Xb, y), 0)
        else:
            import lightgbm as lgb
            b = lgb.LGBMRegressor(objective="l1", n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10, random_state=0, deterministic=True, verbose=-1).fit(Xf, y)
            fitted[m] = (b.predict, 1)
    rows = []
    for split_name, ts_ in (("backtest", te), ("forecast", last)):
        if not len(ts_):
            continue
        hist_ok = ok_hist(ts_)
        Xb, Xf = _features(g, ts_)
        preds = {"persistence": c[ts_], "moving_avg_5s": np.array([np.nanmean(c[max(0, t - 4):t + 1]) for t in ts_])}
        stat = {k: np.where(np.isnan(v), "gap_in_history", "ok") for k, v in preds.items()}
        for m, f in fitted.items():
            if f is None:
                preds[m], stat[m] = np.full(len(ts_), np.nan), np.full(len(ts_), "insufficient_history", dtype=object)
            else:
                v = c[ts_] + f[0](Xf if f[1] else Xb)
                preds[m], stat[m] = np.where(hist_ok, v, np.nan), np.where(hist_ok, "ok", "gap_in_history")
        actual = c[ts_ + H] if split_name == "backtest" else np.full(len(ts_), np.nan)
        for m in models:
            for i, t in enumerate(ts_):
                rows.append({"zone_id": g.zone.iloc[0], "model": m, "split": split_name, "last_observed_sec": int(sec[t]), "target_sec": int(sec[t] + H),
                             "value": float(preds[m][i]), "actual": float(actual[i]), "status": str(stat[m][i])})
    return rows


def run_video_forecast(source, out_dir=None, horizon_s=10, models=MODELS, base_time=None, job_id=None, anomaly=True):
    """영상 CSV -> 예측 결과 CSV(forecast_video.csv)와 이상 이벤트 CSV(anomaly_events.csv).
    base_time: 촬영 시작 시각(있으면 *_at 컬럼을 채움). 반환: {"forecast_path", "anomaly_path", "forecasts", "anomalies", "info"}"""
    bad = set(models) - set(MODELS)
    if bad:
        raise ValueError(f"모르는 모델: {sorted(bad)} (가능: {MODELS})")
    if horizon_s < 1:
        raise ValueError("horizon_s는 1초 이상이어야 합니다.")
    ts, info = load_video_series(source)
    rows = [r for _, g in ts.groupby("zone") for r in forecast_zone(g, horizon_s, models)]
    fc = pd.DataFrame(rows, columns=[c for c in COLUMNS if not c.endswith("_at")])
    bt = pd.Timestamp(base_time, tz="Asia/Seoul") if base_time and pd.Timestamp(base_time).tzinfo is None else (pd.Timestamp(base_time) if base_time else None)
    for a, b in (("last_observed_at", "last_observed_sec"), ("target_at", "target_sec")):
        fc[a] = bt + pd.to_timedelta(fc[b], unit="s") if bt is not None else pd.NaT
    fc["value"], fc["actual"] = fc.value.round(2), fc.actual.round(2)
    fc = fc[COLUMNS]
    out_dir = Path(out_dir) if out_dir else (Path(source) if Path(source).is_dir() else Path(source).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / "forecast_video.csv"
    w = fc.copy()
    for a in ("last_observed_at", "target_at"):
        w[a] = w[a].map(lambda x: x.isoformat() if pd.notna(x) else "")
    w.to_csv(fpath, index=False, encoding="utf-8-sig")
    an, apath, an_info = pd.DataFrame(), None, {}
    if anomaly:
        jb, btime = job_id or Path(info["crowd_csv"]).parent.name, (bt or pd.Timestamp("2026-01-01 00:00", tz="Asia/Seoul")).isoformat()
        an = av.rule_events(ts, job_id=jb, base_time=btime)
        nwin = av.window_features(ts).groupby("zone").size()
        if ts.motion.notna().all() and (nwin >= MIN_IF_WINDOWS).all():
            an = pd.concat([an, av.iforest_events(ts, job_id=jb, base_time=btime)], ignore_index=True)
            an_info["iforest"] = "ok"
        else:
            an_info["iforest"] = f"건너뜀: 구역별 윈도우 {nwin.to_dict()}개(최소 {MIN_IF_WINDOWS}) 또는 flow.csv(motion) 없음 -> 규칙 기반만"
        if bt is None:
            an["event_at"] = pd.NaT  # 촬영 시작 시각을 모르면 event_sec만 의미 있음
        apath = out_dir / "anomaly_events.csv"
        an.assign(event_at=an.event_at.map(lambda x: x.isoformat() if pd.notna(x) else "")).to_csv(apath, index=False, encoding="utf-8-sig")
    bk = fc[fc.split == "backtest"]
    mae = {m: round(float((g.value - g.actual).abs().mean()), 3) if g.status.eq("ok").any() else None for m, g in bk.groupby("model")}
    info.update({"horizon_s": horizon_s, "rows": len(fc), "backtest_mae": mae, "status_counts": fc.groupby(["model", "status"]).size().unstack(fill_value=0).to_dict("index"),
                 "anomaly_events": len(an), **an_info,
                 "note": "학습 필요 모델은 이 영상의 앞 60%로 학습. 영상이 짧으면 해당 모델은 insufficient_history. 짧은 영상의 MAE는 참고용."})
    return {"forecast_path": str(fpath), "anomaly_path": str(apath) if apath else None, "forecasts": fc, "anomalies": an, "info": info}


def main():
    ap = argparse.ArgumentParser(description="영상 분석 CSV로 구역 인원 단기 예측과 이상 흐름 탐지를 실행한다.")
    ap.add_argument("source", help="영상 분석 결과 폴더 또는 crowd_observations.csv")
    ap.add_argument("--out-dir"); ap.add_argument("--horizon", type=int, default=10, help="몇 초 뒤를 예측할지(기본 10)")
    ap.add_argument("--base-time", help="촬영 시작 시각 (예: 2026-10-31 21:00)")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--no-anomaly", action="store_true")
    a = ap.parse_args()
    res = run_video_forecast(a.source, a.out_dir, a.horizon, tuple(a.models), a.base_time, anomaly=not a.no_anomaly)
    i = res["info"]
    print(f"예측: {res['forecast_path']} ({i['rows']}행) | 이상 이벤트: {res['anomaly_path']} ({i['anomaly_events']}건) | 영상 {i['seconds']}초, 구역 {i['zones']}, {i['horizon_s']}초 뒤 예측")
    print("백테스트 MAE(명):", i["backtest_mae"]); print("Isolation Forest:", i.get("iforest"))
    print(pd.DataFrame(i["status_counts"]).T.to_string())


if __name__ == "__main__":
    main()
