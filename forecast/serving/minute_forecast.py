"""영상 CSV -> 구역별 5·10·15분 뒤 인원·밀도 예측 + 경보 (미리 학습한 LightGBM + 그 영상으로 보정).

모델: JuPedSim 시뮬레이션(직선·병목·한산 시나리오, data-area forecast/bottleneck/sim_serving.py train)으로 학습해
      artifacts/minutes/ 에 저장한 LightGBM. 서비스에서는 학습하지 않고 불러와 쓴다.
      - 기준 모델: 공지 일정 없이 학습. 공지 모델: 학습 몰림의 20%를 '공지 안 됨'으로 숨기고 공지 일정 특징까지 학습.
      - 구역에 공지된 몰림(schedule.events)이 있으면 공지 모델, 없으면 기준 모델.
      - 유입 규모가 학습 범위 하위 1%보다 작으면(in_range=False) 모델 대신 직전값.
영상 보정: 예측 시각 + H가 지나 실제 값이 생긴 지난 예측들로, 모델 변화량을 얼마나 믿을지 w(0~1)를 고른다(직전값 쪽으로 줄이기).
      지난 예측이 MIN_MATURED개 모일 때까지는 더 짧은 예측 시간의 w(5분 -> 10분 -> 15분), 그것도 없으면 w=1.
      그 시각까지의 영상만 쓴다(미래 미사용). 시뮬레이션과 실제 영상의 차이(11-20 광화문: 모델이 한산한 구역에 몇 명씩 늘어난다고 봄)를 줄이는 장치.
입력: run_zone_pipeline 결과 폴더(crowd_observations.csv, flow.csv, 있으면 individual_directions.csv) + 구역 정보(zone_meta):
      {구역 이름: {area_m2, width_m, exit_width_m, exit_angle_deg(출구 쪽 이동 방향, 화면 각도), shape(straight|bottleneck),
                  polygon(픽셀, 사람 속도 계산용), schedule(선택: {base_rate 명/초, events [[정점 영상초, 퍼짐 σ초, 정점 추가 유입 명/초], ...]})}}
      사람 속도(m/s) = 박스 높이를 키 1.7m로 보고 환산(detections_csv가 있을 때).
출력: forecast_minutes.csv, alarms_minutes.csv(예측 밀도 >= 3명/㎡ 주의, 5명/㎡ 위험 — 행안부 2024 실외 기준).
최소 영상 길이: 15분(입력 창). 연구용 참고 지표이며 구역 실면적이 틀리면 밀도·경보도 틀린다.
"""
import json, zlib
from pathlib import Path

import lightgbm as lgb
import numpy as np, pandas as pd
from scipy.special import ndtr

from ..anomaly_levels import _in_poly

MODEL_DIR = Path(__file__).resolve().parent / "artifacts" / "minutes"
H = (300, 600, 900)
STEP, HIST = 10, 90  # 10초 칸, 입력 15분 — 학습(sim_models)과 같아야 함
CAP = 1.3  # 출구 용량 명/m/s (학습과 같은 값)
RANGE_COLS = ("in_cap", "in_5m")
LEVELS = ((5.0, "위험"), (3.0, "주의"))
BODY_M, WIN_FRAMES = 1.7, 5
MIN_MATURED, W_GRID = 12, (0.0, 0.25, 0.5, 0.75, 1.0)
FC_COLUMNS = ["zone_id", "last_observed_sec", "horizon_min", "target_sec", "model", "in_range", "correction_w", "n_now", "value",
              "density_now", "density", "actual", "status"]
ALARM_COLUMNS = ["zone_id", "last_observed_sec", "level", "threshold", "first_horizon_min", "pred_density", "density_now", "model", "message"]


def person_speed(folder, detections_csv, meta, fps):
    """구역·초별 사람 평균 속도(m/s). 박스 하단 중앙이 구역 다각형 안인 사람만."""
    ind = pd.read_csv(Path(folder) / "individual_directions.csv")
    m = ind[ind.reliable.astype(bool)].merge(pd.read_csv(detections_csv), on=["frame", "track_id"])
    pts = np.column_stack([(m.x1 + m.x2) / 2, m.y2])
    m["v"] = m.speed_px / WIN_FRAMES * fps / (m.y2 - m.y1).clip(lower=1) * BODY_M
    m["sec"] = (m.frame / fps).astype(int)
    out = [m[_in_poly(pts, v["polygon"])].groupby("sec").v.mean().rename("speed").reset_index().assign(zone=z)
           for z, v in meta.items() if "polygon" in v]
    return pd.concat(out) if out else pd.DataFrame(columns=["sec", "speed", "zone"])


def video_bins(folder, meta, detections_csv=None, fps=10.0):
    """영상 CSV -> 구역별 10초 칸(학습 데이터와 같은 열). 빠진 초가 있어도 칸 번호는 연속으로 맞춘다."""
    folder = Path(folder)
    s = pd.read_csv(folder / "crowd_observations.csv")
    if (folder / "flow.csv").exists():
        s = s.merge(pd.read_csv(folder / "flow.csv")[["sec", "zone", "angle_deg"]], on=["sec", "zone"], how="left")
    else:
        s["angle_deg"] = np.nan
    s = s[s.zone.isin(meta)].copy()
    s["heading"] = np.cos(np.radians(s.angle_deg - s.zone.map({z: v["exit_angle_deg"] for z, v in meta.items()})))
    if detections_csv and (folder / "individual_directions.csv").exists():
        s = s.merge(person_speed(folder, detections_csv, meta, fps), on=["sec", "zone"], how="left")
    else:
        s["speed"] = np.nan  # 속도 모름 -> LightGBM 결측 처리
    s["b"] = s.sec // STEP
    g = s.groupby(["zone", "b"]).agg(n=("count", "last"), inflow=("inflow", "sum"), outflow=("outflow", "sum"),
                                     speed=("speed", "mean"), heading=("heading", "mean"))
    full = pd.MultiIndex.from_tuples([(z, b) for z, d in g.groupby(level=0) for b in range(d.index.get_level_values(1).min(), d.index.get_level_values(1).max() + 1)],
                                     names=["zone", "b"])
    g = g.reindex(full).reset_index()
    g["n"] = g.groupby("zone").n.ffill()  # 빠진 칸 인원 = 직전 값, 유입·유출 = 0
    g[["inflow", "outflow"]] = g[["inflow", "outflow"]].fillna(0)
    g["heading"] = g.heading.fillna(0)
    return g


def _plan(sch, t, cap):
    """공지 일정 특징(학습 sim_models.schedule_features와 같은 식, 오차는 섞지 않음 — 실제 공지는 이미 오차를 담고 있다)."""
    base, bumps = float(sch.get("base_rate", 0.0)), [tuple(map(float, e)) for e in sch.get("events", [])]
    out = {f"plan_in_{h // 60}m": np.full(len(t), base * h) + sum(hh * w * np.sqrt(2 * np.pi) * (ndtr((t + h - c) / w) - ndtr((t - c) / w))
                                                                   for c, w, hh in bumps) for h in H}
    nxt, nh = np.full(len(t), 3600.0), np.zeros(len(t))
    for c, w, hh in sorted(bumps):
        m = (c > t) & (nxt == 3600.0)
        nxt[m], nh[m] = c - t[m], hh / cap
    out["plan_next_in"], out["plan_next_h"] = nxt, nh
    return out


def features(d, z_meta):
    """한 구역의 10초 칸 -> 칸마다(입력 15분이 찬 뒤) 학습과 같은 특징. 반환 (특징 표, 지금 인원, 예측 시각 초, 칸 인원 전체)."""
    n, inf, outf = d.n.to_numpy(float), d.inflow.to_numpy(float), d.outflow.to_numpy(float)
    sp, hd = d.speed.to_numpy(float), d.heading.to_numpy(float)
    a, W, ew, shp = float(z_meta["area_m2"]), float(z_meta["width_m"]), float(z_meta["exit_width_m"]), z_meta.get("shape", "straight")
    cap = CAP * ew
    idx = np.arange(HIST - 1, len(d))
    t = (d.b.to_numpy()[idx] + 1) * STEP
    rows = []
    for i in idx:
        w = lambda x, k: x[i - k + 1:i + 1].sum()
        f = {"n": n[i], "density": n[i] / a, "speed": sp[i], "speed_1m": np.nanmean(sp[i - 5:i + 1]) if np.isfinite(sp[i - 5:i + 1]).any() else np.nan,
             "speed_5m": np.nanmean(sp[i - 29:i + 1]) if np.isfinite(sp[i - 29:i + 1]).any() else np.nan, "heading": hd[i]}
        for k, nm in ((6, "1m"), (30, "5m"), (90, "15m")):
            f[f"in_{nm}"], f[f"out_{nm}"] = w(inf, k), w(outf, k)
            f[f"dn_{nm}"] = n[i] - n[i - k + 1]
        f["in_trend"] = w(inf, 30) - inf[i - 59:i - 29].sum()
        f["in_right_share_5m"] = 0.0  # 영상 구역은 한 방향 유입으로 본다(T자 좌·우 구분 없음)
        f.update(is_bottleneck=float(shp == "bottleneck"), is_tjunction=0.0, width=W, exit_width=ew, entry_width=W, area=a, cap=cap,
                 in_cap=f["in_5m"] / 300 / cap)
        rows.append(f)
    F = pd.DataFrame(rows)
    if len(F):
        for k, v in _plan(z_meta.get("schedule") or {}, t.astype(float), cap).items():
            F[k] = v
    return F, n[idx], t, n


def correct(t, n0, delta, actual, h, fallback=None):
    """걷기식 보정: 시각 t_i에는 t_j + h <= t_i 인 지난 예측만 보고 w를 고른다. 반환 w 배열.
    자기 정답이 MIN_MATURED개 모이기 전에는 fallback(더 짧은 예측 시간에서 고른 w)을 쓴다 — 15분 뒤 정답은 영상 30분 뒤에야 생기므로."""
    w = np.ones(len(t)) if fallback is None else np.asarray(fallback, float).copy()
    for i in range(len(t)):
        m = (t + h <= t[i]) & np.isfinite(actual)
        if m.sum() >= MIN_MATURED:
            w[i] = min(W_GRID, key=lambda x: np.abs(n0[m] + x * delta[m] - actual[m]).mean())
    return w


def run_minute_forecast(folder, zone_meta, detections_csv=None, fps=10.0, out_dir=None, model_dir=MODEL_DIR):
    """-> {"forecasts", "alarms", "forecast_path", "alarm_path", "info"}. zone_meta: dict 또는 JSON 경로."""
    meta = json.loads(Path(zone_meta).read_text(encoding="utf-8")) if isinstance(zone_meta, (str, Path)) else zone_meta
    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    cols, use, p01 = cfg["features"], cfg["use_model"], cfg["train_p01"]
    boosters = {(kind, h): lgb.Booster(model_file=str(Path(model_dir) / f"{kind}_{h // 60}m.txt")) for kind in ("base", "sched") for h in H}
    g = video_bins(folder, meta, detections_csv, fps)
    fc, al, short = [], [], []
    for z, d in g.groupby("zone"):
        F, n0, t, n_all = features(d.sort_values("b").reset_index(drop=True), meta[z])
        if not len(F):
            short.append(z); continue
        a = float(meta[z]["area_m2"])
        sched = bool((meta[z].get("schedule") or {}).get("events"))
        kind = "sched" if sched else "base"
        in_range = np.all([F[c].to_numpy() >= p01[c] for c in RANGE_COLS], 0)
        name = np.where(in_range, "공지" if sched else "기준", "직전값(학습 범위 밖)")
        b0 = d.b.min()
        dens_pred, w_prev = {}, None
        for h in H:
            k = f"{h // 60}m"
            delta = boosters[(kind, h)].predict(F[cols[kind]]) if use[kind][k] else np.zeros(len(F))
            delta = np.where(in_range, delta, 0.0)
            j = (t + h) // STEP - 1 - b0  # 정답 칸 위치
            actual = np.where(j < len(n_all), n_all[np.minimum(j, len(n_all) - 1)], np.nan)
            w = w_prev = correct(t.astype(float), n0, delta, actual, h, w_prev)
            val = np.clip(n0 + w * delta, 0, None)
            dens_pred[h] = val / a
            fc.append(pd.DataFrame({"zone_id": z, "last_observed_sec": t, "horizon_min": h // 60, "target_sec": t + h, "model": name,
                                    "in_range": in_range, "correction_w": w, "n_now": n0, "value": val.round(2), "density_now": (n0 / a).round(4),
                                    "density": (val / a).round(4), "actual": actual,
                                    "status": np.where(np.isfinite(actual), "backtest", "forecast")}))
        for i in range(len(t)):
            for thr, lvl in LEVELS:
                hit = [h for h in H if dens_pred[h][i] >= thr]
                if hit:
                    al.append({"zone_id": z, "last_observed_sec": int(t[i]), "level": lvl, "threshold": thr, "first_horizon_min": hit[0] // 60,
                               "pred_density": round(float(dens_pred[hit[0]][i]), 2), "density_now": round(float(n0[i] / a), 3), "model": name[i],
                               "message": f"{z}: {hit[0] // 60}분 뒤 {dens_pred[hit[0]][i]:.1f}명/㎡ 예상 ({lvl} 기준 {thr:g}명/㎡)"})
                    break
    fc = pd.concat(fc, ignore_index=True)[FC_COLUMNS] if fc else pd.DataFrame(columns=FC_COLUMNS)
    al = pd.DataFrame(al, columns=ALARM_COLUMNS)
    out = Path(out_dir or folder); out.mkdir(parents=True, exist_ok=True)
    fpath, apath = out / "forecast_minutes.csv", out / "alarms_minutes.csv"
    fc.to_csv(fpath, index=False, encoding="utf-8-sig"); al.to_csv(apath, index=False, encoding="utf-8-sig")
    bt = fc[fc.status == "backtest"]
    skill = {}
    for (z, hm), d in bt.groupby(["zone_id", "horizon_min"]):
        e, e0 = (d.value - d.actual).abs().mean(), (d.n_now - d.actual).abs().mean()
        skill[f"{z}/{hm}분"] = {"MAE": round(float(e), 3), "MAE_persistence": round(float(e0), 3), "n": len(d),
                               "model_share": round(float((d.model != "직전값(학습 범위 밖)").mean()), 2)}
    info = {"model_dir": str(model_dir), "model_version": cfg.get("out"), "zones_too_short": short, "rows": len(fc), "alarms": len(al),
            "backtest": skill, "note": f"입력 {HIST * STEP // 60}분이 찬 뒤부터 예측. 구역 실면적·폭이 가정값이면 밀도·경보도 가정이다."}
    return {"forecasts": fc, "alarms": al, "forecast_path": str(fpath), "alarm_path": str(apath), "info": info}
