"""영상 분석 CSV -> 전처리 -> 구역 인원 단기 예측 + 이상 흐름 3단계 -> 결과 CSV.
입력: 팀 영상 파이프라인(backend/app/services/pipeline.py run_zone_pipeline)이 만든 폴더 또는 crowd_observations.csv 경로.
      crowd_observations.csv(sec, zone, count, inflow, outflow) 필수, 같은 폴더의 flow.csv(motion), bottleneck_scores.csv(병목 점수),
      individual_directions.csv(사람별 방향)는 있으면 사용. 역방향·체류 특징은 detections_csv + zones를 함께 넘기면 계산한다.

Python:
    from forecast.serving import run_video_forecast
    res = run_video_forecast("분석결과폴더", horizons=(10, 30, 60), base_time="2026-10-31 21:00",
                             detections_csv="track/xxx_detections.csv", zones=zones, fps=24)
    res["forecast_path"], res["anomaly_path"], res["timeline_path"], res["forecasts"], res["anomalies"], res["timeline"], res["info"]
CLI (저장소 루트에서): python -m forecast.serving.video_pipeline 폴더 [--horizons 10 30 60] [--target mean10|raw] [--base-time ...]

예측 대상(target):
  mean10 (기본) : 10초 평균 인원. H초 뒤 = (t+H-9 ~ t+H초) 평균. 초 단위 탐지 잡음을 줄인 값(명세서 crowd_obs_10s와 같은 집계)
  raw           : 1초 원시 인원
모델 (걷기식: 매초 그 시각까지의 영상만으로 학습·예측, 미래 값 미사용):
  persistence    : 지금 값 그대로 (학습 없음, 기준선)
  moving_avg_5s  : 최근 5초 값 평균 (학습 없음)
  ridge_ar       : 최근 변화 + 유입·유출 + 다른 구역 유입 -> H초 뒤 변화 (매초 다시 학습, 강한 규제, 학습 행 30개 이상)
  lightgbm       : 같은 특징 + 다른 구역 인원 변화·이동량·역방향·체류·도착·이탈 예정 (30초마다 다시 학습, 학습 행 200개 이상 ≈ 4분 이상 영상,
                   잎 7·잎당 최소 80행의 단순한 설정 LGB_WALK_PARAMS)
미리 학습 모델 (쌓인 영상 여러 개로 forecast/training/train_video.py가 학습해 artifacts/video/에 저장. 서비스에서는 학습 없이 첫 초부터 예측):
  lightgbm_pre   : 표 특징(pre_design). 결과의 shap_top = 이 예측에 크게 기여한 특징 3개(LightGBM 내장 TreeSHAP, 사람 수 단위)
  lstm_pre       : 최근 60초 시퀀스. 예측 시간 여러 개를 한 모델로 동시에 낸다
  카메라마다 인원 규모가 달라 특징·정답을 '그 구역 지금까지 평균 인원(ref)'으로 나눈 비율로 쓴다. 모델 파일이 없으면 status=no_pretrained_model.
평가: split=backtest(실제 있음)에서 MAE와 직전값 대비 개선율 skill = 1 - MAE(모델)/MAE(persistence). split=forecast는 영상 끝 + H초.
연구용 참고 지표이며 현장 상황 판단은 운영자 확인이 필요하다."""
import argparse
from functools import lru_cache
from pathlib import Path
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .. import anomaly_video as av
from .. import anomaly_levels as al
from .minute_forecast import run_minute_forecast

P = 5  # 입력으로 쓰는 과거 시점 수
REFIT_EVERY_S = 30  # LightGBM 재학습 간격(초). ridge는 매초 다시 학습
MIN_TRAIN = {"ridge_ar": 30, "lightgbm": 200}  # ridge 특징 8개 -> 최소 30행
RIDGE_ALPHA = 10.0  # 학습 행이 적어 강하게 규제(과적합 방지)
MIN_IF_WINDOWS = 30  # 명세서 TC-ANO-04: 윈도우 29개는 Isolation Forest 학습 불가, 30개부터 가능
MODELS = ("persistence", "moving_avg_5s", "ridge_ar", "lightgbm", "lightgbm_pre", "lstm_pre")
TARGETS = ("mean10", "raw")
COLUMNS = ["zone_id", "model", "horizon_s", "target", "split", "last_observed_sec", "target_sec", "last_observed_at", "target_at", "value", "actual", "status", "shap_top"]
PRE_DIR = Path(__file__).resolve().parent / "artifacts" / "video"  # 미리 학습 모델 위치(train_video.py가 씀)
ARRIVAL_COLS = ["arr10", "arr30", "arr60", "dep10", "dep30", "dep60", "net10", "net30", "net60"]  # 도착·이탈 예정(anomaly_levels.arrival_features)
PRE_FEATURES = ["level_ratio", "d1", "d2", "d3", "d4", "in5", "out5", "oth_in5", "oth_dcnt5", "motion5", "reverse_ratio", "dwell"] + ARRIVAL_COLS
LGB_WALK_PARAMS = dict(objective="l1", n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=80, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.6, reg_lambda=1.0, random_state=0, deterministic=True, verbose=-1)  # 'D' 설정: 행이 적어 단순하게(광화문 실험)
SEQ_L = 60  # LSTM 입력 시퀀스 길이(초). 영상 초반은 앞을 0으로 채우고 valid 채널로 표시
SEQ_FEATURES = ["valid", "level_ratio", "in5", "out5", "oth_in5", "oth_dcnt5", "motion5", "rel_to_now"]


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


ARRIVAL_CSV, TRAJ_CSV = "arrival_features.csv", "trajectory_features.csv"


def attach_traj(ts, tf):
    """구역 시계열에 역방향 비율·체류 비율(anomaly_levels.trajectory_features)을 붙인다. 없는 초는 0."""
    return ts.merge(tf[["sec", "zone", "reverse_ratio", "dwell"]], on=["sec", "zone"], how="left").fillna({"reverse_ratio": 0.0, "dwell": 0.0})


def attach_arrival(ts, af):
    """구역 시계열에 도착·이탈 예정(arrH, depH, netH = arrH - depH)을 붙인다. 없는 초는 0."""
    af = af.copy()
    for h in (10, 30, 60):
        af[f"net{h}"] = af[f"arr{h}"] - af[f"dep{h}"]
    return ts.merge(af[["sec", "zone"] + ARRIVAL_COLS], on=["sec", "zone"], how="left").fillna({c: 0.0 for c in ARRIVAL_COLS})


def _target_series(c, target):
    """예측 대상 시계열. mean10 = 지금까지 10초 평균(과거만), raw = 원시 인원."""
    s = pd.Series(c)
    return s.rolling(10, min_periods=10).mean().to_numpy() if target == "mean10" else s.to_numpy(float)


def _zone_features(ts):
    """구역별 특징 표(sec 기준). 자기 구역 최근 5초 유입·유출, 다른 구역 최근 5초 유입 합·인원 변화(앞서가는 신호), 이동량, 역방향, 체류."""
    roll = lambda s, f: s.rolling(P, min_periods=1).agg(f)
    piv = {k: ts.pivot(index="sec", columns="zone", values=k) for k in ("count", "inflow", "outflow")}
    feats = {}
    for z in ts.zone.unique():
        g = ts[ts.zone == z].set_index("sec")
        others = [o for o in piv["count"].columns if o != z]
        f = pd.DataFrame(index=g.index)
        f["in5"], f["out5"] = roll(g.inflow.fillna(0), "sum"), roll(g.outflow.fillna(0), "sum")
        f["oth_in5"] = roll(piv["inflow"][others].fillna(0).sum(1), "sum").reindex(g.index) if others else 0.0
        f["oth_dcnt5"] = (piv["count"][others].sum(1) - piv["count"][others].sum(1).shift(P)).reindex(g.index) if others else 0.0
        f["motion5"] = roll(g.motion, "mean")
        for c in ("reverse_ratio", "dwell"):
            f[c] = roll(g[c], "mean") if c in g else np.nan
        for c in ARRIVAL_COLS:  # 사람 수 단위. detections가 없으면 0
            f[c] = g[c] if c in g else 0.0
        feats[z] = f.fillna(0.0)
    return feats


def _ridge(X, y, alpha=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = (X - mu) / sd
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (y - y.mean()))
    return lambda Z: ((Z - mu) / sd) @ w + y.mean()


def pre_design(g, extra, target):
    """미리 학습 모델의 입력(학습·서비스 공용, 과거 값만). 카메라·구역마다 규모가 달라 ref(그 구역 t까지 평균 인원, 최소 1)로 나눈다.
    반환 dict: X[행, PRE_FEATURES](LightGBM), S[행, SEQ_L, SEQ_FEATURES](LSTM), y, ref, ok(입력 유효), ts_all(위치 인덱스).
    정답(학습용) = (y[t+H] - y[t]) / ref."""
    g = g.reset_index(drop=True)
    y = _target_series(g["count"].to_numpy(float), target)
    ts_all = np.arange(P - 1, len(y))
    lag = np.column_stack([y[ts_all - k] for k in range(P)]) if len(ts_all) else np.zeros((0, P))
    ref_all = np.fmax(pd.Series(y).expanding().mean().to_numpy(), 1.0)
    r = ref_all[ts_all]
    ex = extra.reindex(g.sec.to_numpy()).to_numpy(float)  # in5, out5, oth_in5, oth_dcnt5 (인원) | motion5, reverse_ratio, dwell | 도착 예정 9개(인원)
    X = np.column_stack([lag[:, 0] / r] + [(lag[:, k] - lag[:, 0]) / r for k in range(1, P)]
                        + [ex[ts_all, :4] / r[:, None], ex[ts_all, 4:7], ex[ts_all, 7:] / r[:, None]])
    M = np.column_stack([~np.isnan(y), np.nan_to_num(y), ex[:, :5]]).astype(np.float32)
    M = np.vstack([np.zeros((SEQ_L - 1, M.shape[1]), np.float32), M])
    S = np.transpose(sliding_window_view(M, SEQ_L, axis=0)[ts_all], (0, 2, 1)).copy()  # 행 t = 원래 t-SEQ_L+1 ~ t
    S[:, :, 1:6] /= r[:, None, None].astype(np.float32)  # 인원 단위 채널만 ref로 나눔(motion5는 그대로)
    rel = (S[:, :, 1] - S[:, -1:, 1]) * S[:, :, 0]       # 지금 대비 변화(추세 모양). 채운 앞부분은 0
    S = np.concatenate([S, rel[:, :, None]], axis=2)
    return {"X": X, "S": S, "y": y, "ref": r, "ok": ~np.isnan(lag).any(1), "ts_all": ts_all}


@lru_cache(maxsize=16)
def _load_pre(path, mtime, kind):  # mtime을 키에 넣어 재학습으로 파일이 바뀌면 서버 재시작 없이 다시 읽는다
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.Booster(model_file=path)
    import torch
    from .video_lstm import VideoNet
    ck = torch.load(path, map_location="cpu")
    net = VideoNet(ck["n_in"], ck["hidden"], len(ck["horizons"]))
    net.load_state_dict(ck["state"])
    return net.eval(), [int(h) for h in ck["horizons"]]


def pre_models(target, pre_dir=None):
    """-> {"lgbm": {H: Booster}, "lstm": (net, horizons) 또는 None}"""
    d = Path(pre_dir or PRE_DIR)
    load = lambda p, kind: _load_pre(str(p), p.stat().st_mtime, kind) if p.exists() else None
    return {"lgbm": {int(p.stem.rsplit("_", 1)[1][:-1]): load(p, "lgbm") for p in d.glob(f"lgbm_{target}_*s.txt")},
            "lstm": load(d / f"lstm_{target}.pt", "lstm")}


def _shap_text(c, names=PRE_FEATURES, k=3):
    top = np.argsort(-np.abs(c))[:k]
    return ", ".join(f"{names[i]} {c[i]:+.2f}명" for i in top)


def _pre_predict(g, extra, H, models, target, y, ts_all, pre_dir):
    """미리 학습 모델 예측 -> (preds, stat, shap_top). 모델 파일이 없거나 H를 학습하지 않았으면 no_pretrained_model."""
    preds, stat, shap = {}, {}, None
    pm, d = pre_models(target, pre_dir), pre_design(g, extra, target)
    base, r, ok = y[ts_all], d["ref"], d["ok"]
    none = lambda: (np.full(len(ts_all), np.nan), np.full(len(ts_all), "no_pretrained_model", dtype=object))
    okst = lambda: np.where(ok, "ok", "insufficient_history").astype(object)
    if "lightgbm_pre" in models:
        b = pm["lgbm"].get(H)
        if b is None or b.num_feature() != d["X"].shape[1]:  # 없거나 특징 수가 다른 옛 모델(도착 예정 특징 추가 전)
            preds["lightgbm_pre"], stat["lightgbm_pre"] = none()
        else:
            contrib = b.predict(d["X"], pred_contrib=True)  # 마지막 열 = 기준값, 합 = 예측(비율)
            preds["lightgbm_pre"], stat["lightgbm_pre"] = np.where(ok, base + contrib.sum(1) * r, np.nan), okst()
            shap = [_shap_text(c[:-1] * ri) if o else "" for c, ri, o in zip(contrib, r, ok)]
    if "lstm_pre" in models:
        net = pm["lstm"]
        if net is None or H not in net[1]:
            preds["lstm_pre"], stat["lstm_pre"] = none()
        else:
            from .video_lstm import lstm_predict
            out = lstm_predict(net[0], d["S"])[:, net[1].index(H)]
            preds["lstm_pre"], stat["lstm_pre"] = np.where(ok, base + out * r, np.nan), okst()
    return preds, stat, shap


def forecast_zone(g, extra, H, models=MODELS, target="mean10", refit_every=REFIT_EVERY_S, pre_dir=None):
    """한 구역, 한 예측 시간. 매초 t에서 t까지의 값만으로 y(t+H)를 예측한다(확장 창 학습, 정답 시각이 t 이전인 행만 학습)."""
    g = g.reset_index(drop=True); sec = g.sec.to_numpy()
    y = _target_series(g["count"].to_numpy(float), target); n = len(y)
    ts_all = np.arange(P - 1, n)
    lag = np.column_stack([y[ts_all - k] for k in range(P)])
    hist_ok = ~np.isnan(lag).any(1)
    ex = extra.reindex(sec).to_numpy(float)[ts_all]
    Xb = np.column_stack([lag[:, 0]] + [lag[:, k] - lag[:, 0] for k in range(1, P)] + [ex[:, :3]])  # 수준 + 최근 변화 + 유입·유출 + 다른 구역 유입
    Xf = np.column_stack([lag[:, 0]] + [lag[:, k] - lag[:, 0] for k in range(1, P)] + [ex])       # + 다른 구역 인원 변화·이동량·역방향·체류
    pos = {t: i for i, t in enumerate(ts_all)}
    preds = {"persistence": lag[:, 0].copy(), "moving_avg_5s": np.where(hist_ok, np.nan_to_num(lag).mean(1), np.nan)}
    stat = {k: np.where(~hist_ok, "insufficient_history", "ok").astype(object) for k in preds}
    shap = None
    if {"lightgbm_pre", "lstm_pre"} & set(models):
        p2, s2, shap = _pre_predict(g, extra, H, models, target, y, ts_all, pre_dir)
        preds.update(p2); stat.update(s2)
    for m in ("ridge_ar", "lightgbm"):
        if m not in models:
            continue
        p_m, s_m, fit, fit_at = np.full(len(ts_all), np.nan), np.full(len(ts_all), "insufficient_history", dtype=object), None, -10 ** 9
        for i, t in enumerate(ts_all):
            tr = ts_all[(ts_all + H <= t) & hist_ok]
            tr = tr[~np.isnan(y[tr + H])]
            if len(tr) < MIN_TRAIN[m] or not hist_ok[i]:
                continue
            if fit is None or t - fit_at >= (1 if m == "ridge_ar" else refit_every):
                idx = [pos[x] for x in tr]; yy = y[tr + H] - y[tr]
                if m == "ridge_ar":
                    fit = (_ridge(Xb[idx], yy, RIDGE_ALPHA), Xb)
                else:
                    import lightgbm as lgb
                    b = lgb.LGBMRegressor(**LGB_WALK_PARAMS).fit(Xf[idx], yy)
                    fit = (b.predict, Xf)
                fit_at = t
            p_m[i] = y[t] + fit[0](fit[1][i:i + 1])[0]; s_m[i] = "ok"
        preds[m], stat[m] = p_m, s_m
    rows = []
    for i, t in enumerate(ts_all):
        future = t + H >= n
        if future and t != n - 1:
            continue  # 실제가 없는 중간 시각은 건너뛰고 마지막 시각만 운용 예측으로
        for m in models:
            rows.append({"zone_id": g.zone.iloc[0], "model": m, "horizon_s": H, "target": target, "split": "forecast" if future else "backtest",
                         "last_observed_sec": int(sec[t]), "target_sec": int(sec[t] + H), "value": float(preds[m][i]),
                         "actual": float("nan") if future else float(y[t + H]), "status": str(stat[m][i]),
                         "shap_top": shap[i] if m == "lightgbm_pre" and shap is not None else ""})
    return rows


def skill_table(fc):
    """예측 시간·모델별 backtest MAE·RMSE와 직전값 대비 개선율(같은 행에서 비교)."""
    bk = fc[(fc.split == "backtest")]
    out = {}
    for h, g in bk.groupby("horizon_s"):
        per = g.pivot_table(index=["zone_id", "last_observed_sec"], columns="model", values="value")
        act = g.groupby(["zone_id", "last_observed_sec"]).actual.first()
        for m in per.columns:
            ok = per[m].notna() & per["persistence"].notna() & act.notna()
            if not ok.any():
                out[(h, m)] = {"MAE": None, "RMSE": None, "skill_vs_persistence": None, "n": 0}; continue
            err = (per[m] - act)[ok]
            e, e0 = err.abs().mean(), (per["persistence"] - act)[ok].abs().mean()
            out[(h, m)] = {"MAE": round(float(e), 3), "RMSE": round(float(np.sqrt((err ** 2).mean())), 3),
                           "skill_vs_persistence": round(float(1 - e / e0), 3) if e0 > 0 else None, "n": int(ok.sum())}
    return out


def run_video_forecast(source, out_dir=None, horizons=(10, 30, 60), models=MODELS, base_time=None, job_id=None, anomaly=True,
                       target="mean10", detections_csv=None, zones=None, fps=None, horizon_s=None, pre_dir=None, zone_meta=None):
    """영상 CSV -> forecast_video.csv(예측), anomaly_events.csv(이상 이벤트), anomaly_timeline.csv(구역별 매초 3단계).
    base_time: 촬영 시작 시각(있으면 *_at 컬럼을 채움). detections_csv + zones(+fps)를 주면 역방향·체류 특징을 계산한다.
    horizon_s(정수)는 이전 호출 방식 호환용으로, 주면 horizons=(horizon_s,)와 같다. pre_dir: 미리 학습 모델 폴더(기본 artifacts/video).
    zone_meta(구역 실면적·폭·출구 방향 dict 또는 JSON 경로)를 주면 5·10·15분 뒤 예측 + 경보(minute_forecast)도 실행해
    forecast_minutes.csv, alarms_minutes.csv 를 남긴다. horizons=()이면 초 단위 예측은 건너뛴다."""
    if horizon_s is not None:
        horizons = (horizon_s,)
    bad = set(models) - set(MODELS)
    if bad:
        raise ValueError(f"모르는 모델: {sorted(bad)} (가능: {MODELS})")
    if target not in TARGETS:
        raise ValueError(f"target은 {TARGETS} 중 하나")
    if horizons and min(horizons) < 1:
        raise ValueError("예측 시간(horizons)은 1초 이상이어야 합니다.")
    if not horizons and zone_meta is None:
        raise ValueError("horizons(초 단위)나 zone_meta(분 단위) 중 하나는 있어야 합니다.")
    ts, info = load_video_series(source)
    folder = Path(info["crowd_csv"]).parent
    if detections_csv is not None and zones is not None and (folder / "individual_directions.csv").exists():
        if fps is None:  # detections의 t_ms로 fps 추정
            d = pd.read_csv(detections_csv, usecols=["frame", "t_ms"]); d = d[d.t_ms > 0]
            fps = float((d.frame / (d.t_ms / 1000)).median())
        tf = al.trajectory_features(folder / "individual_directions.csv", detections_csv, zones, fps)
        tf.to_csv(folder / TRAJ_CSV, index=False)  # train_video가 같은 특징으로 학습하도록 영상 폴더에 남김
        ts = attach_traj(ts, tf)
        info["trajectory_features"] = f"reverse_ratio, dwell 계산함({TRAJ_CSV} 저장)"
    elif (folder / TRAJ_CSV).exists():
        ts = attach_traj(ts, pd.read_csv(folder / TRAJ_CSV))
        info["trajectory_features"] = f"{TRAJ_CSV} 사용"
    else:
        info["trajectory_features"] = "없음(detections_csv·zones·individual_directions.csv가 있어야 계산)"
    if detections_csv is not None and zones is not None:
        af = al.arrival_features(detections_csv, zones)
        af.to_csv(folder / ARRIVAL_CSV, index=False)  # 나중에 train_video가 같은 특징으로 학습하도록 영상 폴더에 남김
        ts = attach_arrival(ts, af)
        info["arrival_features"] = f"도착·이탈 예정 계산함({ARRIVAL_CSV} 저장)"
    elif (folder / ARRIVAL_CSV).exists():
        ts = attach_arrival(ts, pd.read_csv(folder / ARRIVAL_CSV))
        info["arrival_features"] = f"{ARRIVAL_CSV} 사용"
    else:
        info["arrival_features"] = "없음(detections_csv·zones가 있어야 계산, 0으로 채움)"
    extra = _zone_features(ts)
    rows = [r for H in horizons for z, g in ts.groupby("zone") for r in forecast_zone(g, extra[z], H, models, target, pre_dir=pre_dir)]
    fc = pd.DataFrame(rows, columns=[c for c in COLUMNS if not c.endswith("_at")])
    bt = pd.Timestamp(base_time, tz="Asia/Seoul") if base_time and pd.Timestamp(base_time).tzinfo is None else (pd.Timestamp(base_time) if base_time else None)
    for a, b in (("last_observed_at", "last_observed_sec"), ("target_at", "target_sec")):
        fc[a] = bt + pd.to_timedelta(fc[b], unit="s") if bt is not None else pd.NaT
    fc["value"], fc["actual"] = fc.value.round(2), fc.actual.round(2)
    fc = fc[COLUMNS]
    out_dir = Path(out_dir) if out_dir else (Path(source) if Path(source).is_dir() else Path(source).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    iso = lambda s: s.map(lambda x: x.isoformat() if pd.notna(x) else "")
    fpath = out_dir / "forecast_video.csv"
    fc.assign(last_observed_at=iso(fc.last_observed_at), target_at=iso(fc.target_at)).to_csv(fpath, index=False, encoding="utf-8-sig")
    an, tl, apath, tpath, an_info = pd.DataFrame(columns=al.EVENT_COLUMNS), pd.DataFrame(), None, None, {}
    if anomaly:
        jb, btime = job_id or folder.name, (bt or pd.Timestamp("2026-01-01 00:00", tz="Asia/Seoul")).isoformat()
        rules = av.rule_events(ts, job_id=jb, base_time=btime)
        nwin = av.window_features(ts).groupby("zone").size()
        sc, ife = None, pd.DataFrame()
        # Optical Flow는 첫 초(이전 프레임 없음)처럼 몇 초가 비어 있을 수 있다. window_features가 빈 칸을 버리므로 구역마다 motion이 있으면 된다.
        has_motion = ts.groupby("zone").motion.apply(lambda s: s.notna().any()).all()
        if has_motion and (nwin >= MIN_IF_WINDOWS).all():
            sc, feats = av.iforest_scores(ts)
            ife = av.iforest_events(ts, job_id=jb, base_time=btime)
            an_info["iforest"] = f"ok (특징 {feats})"
        else:
            an_info["iforest"] = f"건너뜀: 구역별 윈도우 {nwin.to_dict()}개(최소 {MIN_IF_WINDOWS}) 또는 flow.csv(motion) 없음 -> 규칙 기반만"
        bn = pd.read_csv(folder / "bottleneck_scores.csv") if (folder / "bottleneck_scores.csv").exists() else None
        an_info["bottleneck"] = "반영" if bn is not None else "없음(bottleneck_scores.csv 없음)"
        tl = al.grade_levels(ts, rules, sc, bn)
        lv = al.level_events(tl, job_id=jb, base_time=btime)
        an = pd.concat([lv, rules.reindex(columns=al.EVENT_COLUMNS), ife.reindex(columns=al.EVENT_COLUMNS)], ignore_index=True)
        an = an.sort_values(["zone_id", "event_sec", "method"]).reset_index(drop=True)
        if bt is None:
            an["event_at"] = pd.NaT  # 촬영 시작 시각을 모르면 event_sec만 의미 있음
        apath, tpath = out_dir / "anomaly_events.csv", out_dir / "anomaly_timeline.csv"
        an.assign(event_at=iso(an.event_at)).to_csv(apath, index=False, encoding="utf-8-sig")
        tl.to_csv(tpath, index=False, encoding="utf-8-sig")
        an_info["level_seconds"] = tl.groupby(["zone_id", "level"]).size().unstack(fill_value=0).to_dict("index")
    mres = None
    if zone_meta is not None:
        mres = run_minute_forecast(folder, zone_meta, detections_csv, fps or 10.0, out_dir)
        info["minutes"] = mres["info"]
    sk = skill_table(fc)
    info.update({"horizons": list(horizons), "target": target, "rows": len(fc),
                 "backtest": {f"{h}s/{m}": v for (h, m), v in sk.items()},
                 "status_counts": fc.groupby(["horizon_s", "model", "status"]).size().unstack(fill_value=0).to_dict("index"),
                 "anomaly_events": int(len(an)), "level_events": int((an.method == "level").sum()) if len(an) else 0, **an_info,
                 "note": "걷기식 예측: 매초 그 시각까지의 영상만으로 학습·예측. skill>0이면 직전값보다 낫다. 짧은 영상의 수치는 참고용."})
    return {"forecast_path": str(fpath), "anomaly_path": str(apath) if apath else None, "timeline_path": str(tpath) if tpath else None,
            "forecasts": fc, "anomalies": an, "timeline": tl, "info": info,
            "minute_forecast_path": mres["forecast_path"] if mres else None, "minute_alarm_path": mres["alarm_path"] if mres else None,
            "minute_forecasts": mres["forecasts"] if mres else None, "minute_alarms": mres["alarms"] if mres else None}


def main():
    ap = argparse.ArgumentParser(description="영상 분석 CSV로 구역 인원 단기 예측과 이상 흐름 3단계를 실행한다.")
    ap.add_argument("source", help="영상 분석 결과 폴더 또는 crowd_observations.csv")
    ap.add_argument("--out-dir"); ap.add_argument("--horizons", type=int, nargs="*", default=[10, 30, 60], help="몇 초 뒤를 예측할지(값 없이 --horizons 만 쓰면 초 단위 예측 건너뜀)")
    ap.add_argument("--target", default="mean10", choices=TARGETS)
    ap.add_argument("--base-time", help="촬영 시작 시각 (예: 2026-10-31 21:00)")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--no-anomaly", action="store_true")
    ap.add_argument("--pre-dir", help=f"미리 학습 모델 폴더(기본 {PRE_DIR})")
    ap.add_argument("--zone-meta", help="구역 실면적·폭·출구 방향 JSON -> 5·10·15분 예측 + 경보(forecast_minutes.csv, alarms_minutes.csv)")
    ap.add_argument("--detections", help="사람 박스 CSV(분 단위 예측의 사람 속도 계산용)")
    ap.add_argument("--fps", type=float)
    a = ap.parse_args()
    res = run_video_forecast(a.source, a.out_dir, tuple(a.horizons), tuple(a.models), a.base_time, anomaly=not a.no_anomaly, target=a.target,
                             pre_dir=a.pre_dir, zone_meta=a.zone_meta, detections_csv=a.detections, fps=a.fps)
    i = res["info"]
    print(f"예측: {res['forecast_path']} ({i['rows']}행) | 이상 이벤트: {res['anomaly_path']} ({i['anomaly_events']}건, 단계 이벤트 {i['level_events']}) | 영상 {i['seconds']}초, 구역 {i['zones']}")
    if i["backtest"]:
        print(pd.DataFrame(i["backtest"]).T.to_string())
    print("Isolation Forest:", i.get("iforest"), "| 병목:", i.get("bottleneck"))
    if res["minute_forecast_path"]:
        m = i["minutes"]
        print(f"분 단위: {res['minute_forecast_path']} ({m['rows']}행) | 경보: {res['minute_alarm_path']} ({m['alarms']}건) | 모델 {m['model_version']}")
        print(pd.DataFrame(m["backtest"]).T.to_string())


if __name__ == "__main__":
    main()
