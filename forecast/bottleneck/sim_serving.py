"""영상 CSV -> 구역별 5/10/15분 뒤 인원·밀도 예측 + 경보 (JuPedSim 시뮬레이션으로 학습한 LightGBM).

1) 학습·저장:  python -m forecast.bottleneck.sim_serving train --out 20261120_120_serve --hide 0.2
   runs/<out>/models/ 에 기준 모델(공지 특징 없음)과 공지 모델(학습 몰림 hide 비율을 숨겨 학습) 5/10/15분 각각 저장.
2) 예측:      python -m forecast.bottleneck.sim_serving predict --model 20261120_120_serve --zones-dir <run_zone_pipeline 결과> --meta zone_meta.json
   입력: crowd_observations.csv(sec, zone, count, inflow, outflow), flow.csv(sec, zone, angle_deg, ...),
         (있으면) individual_directions.csv + --detections(사람 박스) -> 사람 속도(m/s, 박스 높이 = 키 1.7m로 환산).
   구역 정보(meta JSON, 구역 이름 -> 값): area_m2, width_m, exit_width_m, exit_angle_deg(출구 쪽 이동 방향, 화면 각도), shape(straight|bottleneck),
         polygon(속도 계산용, 픽셀), schedule(선택: base_rate 평소 유입 명/초, events [[정점 시각(영상 초), 퍼짐 σ(초), 정점 추가 유입 명/초], ...]).
   전환 규칙: 구역에 공지된 몰림(events)이 있으면 공지 모델, 없으면 기준 모델.
   시간 구간마다 검증에서 직전값보다 못했던 모델은 쓰지 않고 직전값(지금 인원 유지)을 낸다(config.json use_model).
   유입 규모가 학습 범위 하위 1%보다 작으면(in_range=False) 모델 대신 직전값 — 그때 경보는 지금 밀도가 기준을 넘을 때만.
   출력(--out-dir, 기본 zones-dir): forecast_minutes.csv(10초마다), alarms_minutes.csv(예측 밀도 >= 주의 3명/㎡, 위험 5명/㎡ — 행안부 2024 실외 기준).
한계: 학습은 시뮬레이션(30m 구역, 한 방향 흐름)이다. 실제 카메라 구역의 실면적·폭이 틀리면 밀도와 경보가 그대로 틀린다.
"""
import argparse, json, time
from pathlib import Path

import lightgbm as lgb
import numpy as np, pandas as pd
import shapely

from .sim_models import H, HIST, RUNS, STEP, build, fit_lgbm, load

RANGE_COLS = ("in_cap", "in_5m")  # 유입 규모가 학습(시뮬레이션) 범위 안인지 볼 특징
LEVELS = ((5.0, "위험"), (3.0, "주의"))  # 명/㎡, 높은 것부터
BODY_M = 1.7  # 사람 박스 높이 = 키(m) 가정
WIN_FRAMES = 5  # individual_directions.speed_px 가 몇 프레임 이동량인지(팀 tracker TRACK_WINDOW_FRAMES)


def split_ids(sc):
    rng = np.random.default_rng(0)  # sim_models·sim_announce와 같은 분할
    sids = rng.permutation(sc.sid.to_numpy())
    ntr, nva = int(len(sids) * 0.7), int(len(sids) * 0.15)
    return set(sids[:ntr]), set(sids[ntr:ntr + nva])


def train(out, hide, shapes, drop=()):
    run = RUNS / out
    if run.exists() and any(run.iterdir()):
        raise SystemExit(f"{run} 가 이미 있어요. 다른 --out 이름을 쓰세요.")
    (run / "models").mkdir(parents=True)
    g, sc = load()
    sc = sc[sc["shape"].isin(shapes)]; g = g[g.sid.isin(sc.sid)]
    sci = sc.set_index("sid", drop=False)
    F0, _, Y, _, M = build(g)
    Fp = build(g, sci, hide=hide)[0]
    trs, vas = split_ids(sc)
    tr, va = M.sid.isin(trs).to_numpy(), M.sid.isin(vas).to_numpy()
    ho = ~tr  # 학습에 안 쓴 시나리오 전부(검증 + 테스트): 모델을 쓸지 정하는 기준. 검증 18개만 보면 우연에 흔들림(11-20 확인)
    cols = {"base": [c for c in F0.columns if c not in drop], "sched": [c for c in Fp.columns if c not in drop]}
    use, val = {}, {}
    for kind, F in (("base", F0), ("sched", Fp)):
        _, pv, models = fit_lgbm(F[tr], Y[tr], F[va], Y[va], F[ho], cols[kind], return_models=True)
        for h, m in zip(H, models):
            m.save_model(str(run / "models" / f"{kind}_{h // 60}m.txt"), num_iteration=m.best_iteration)
        # 검증에서 직전값(변화 0)보다 못한 시간 구간은 서비스에서 직전값을 쓴다
        mae, mae0 = np.abs(pv - Y[ho]).mean(0), np.abs(Y[ho]).mean(0)
        use[kind] = {f"{h // 60}m": bool(mae[k] < mae0[k]) for k, h in enumerate(H)}
        val[kind] = {f"{h // 60}m": [round(float(mae[k]), 2), round(float(mae0[k]), 2)] for k, h in enumerate(H)}
    print("학습 밖(검증+테스트) MAE [모델, 직전값]:", val); print("모델 사용:", use)
    rng_ = {c: round(float(np.nanpercentile(F0[c], 1)), 4) for c in RANGE_COLS}  # 학습 범위 하한(1%)
    (run / "config.json").write_text(json.dumps(dict(out=out, script="sim_serving train", hide_train=hide, shapes=shapes, dropped=list(drop), horizons_s=H,
                                                     features=cols, use_model=use, train_p01=rng_, val_mae_model_vs_persist=val, n_scenarios=len(sc), created=time.strftime("%Y-%m-%d %H:%M")),
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {run / 'models'}")


def person_speed(zones_dir, detections, meta, fps):
    """구역·초별 사람 평균 속도(m/s). 박스 하단 중앙이 구역 다각형 안인 사람만."""
    ind = pd.read_csv(Path(zones_dir) / "individual_directions.csv")
    det = pd.read_csv(detections)
    m = ind[ind.reliable.astype(bool)].merge(det, on=["frame", "track_id"])
    bx, by = ((m.x1 + m.x2) / 2).to_numpy(), m.y2.to_numpy()
    m["v"] = m.speed_px / WIN_FRAMES * fps / (m.y2 - m.y1).clip(lower=1) * BODY_M
    m["sec"] = (m.frame / fps).astype(int)
    out = []
    for z, z_meta in meta.items():
        if "polygon" in z_meta:
            inside = shapely.contains_xy(shapely.Polygon(z_meta["polygon"]), bx, by)
            out.append(m[inside].groupby("sec").v.mean().rename("speed").reset_index().assign(zone=z))
    return pd.concat(out) if out else pd.DataFrame(columns=["sec", "speed", "zone"])


def video_bins(zones_dir, meta, detections=None, fps=10.0):
    """영상 CSV -> sim_models.load()와 같은 10초 칸 표(구역 = sid)."""
    zd = Path(zones_dir)
    obs, flow = pd.read_csv(zd / "crowd_observations.csv"), pd.read_csv(zd / "flow.csv")
    s = obs.merge(flow[["sec", "zone", "angle_deg"]], on=["sec", "zone"], how="left")
    s = s[s.zone.isin(meta)]
    s["heading"] = np.cos(np.radians(s.angle_deg - s.zone.map({z: v["exit_angle_deg"] for z, v in meta.items()})))
    if detections and (zd / "individual_directions.csv").exists():
        s = s.merge(person_speed(zd, detections, meta, fps), on=["sec", "zone"], how="left")
    else:
        s["speed"] = np.nan  # 속도 모름: LightGBM이 결측 방향으로 처리
    s["b"] = s.sec // STEP
    g = s.groupby(["zone", "b"]).agg(n=("count", "last"), inflow=("inflow", "sum"), outflow=("outflow", "sum"),
                                     speed=("speed", "mean"), heading=("heading", "mean")).reset_index()
    g["in_right"] = 0.0
    for col, key in (("width", "width_m"), ("exit_width", "exit_width_m"), ("area", "area_m2")):
        g[col] = g.zone.map({z: float(v[key]) for z, v in meta.items()})
    g["entry_width"], g["shape"] = g.width, g.zone.map({z: v.get("shape", "straight") for z, v in meta.items()})
    g["sid"] = g.zone
    g["density"] = g.n / g.area
    g["in_w"], g["out_w"], g["in_r_w"] = g.inflow / g.width, g.outflow / g.width, g.in_right / g.width
    g["heading"] = g.heading.fillna(0)  # speed는 결측 그대로(사람 없음과 모름을 구분 못 하므로 0으로 채우지 않음)
    return g


def schedule_row(zone, z_meta):
    sch = z_meta.get("schedule") or {}
    return {"sid": zone, "base": float(sch.get("base_rate", 0.0)), "bumps": json.dumps(sch.get("events", []))}


def predict(model, zones_dir, meta_path, detections=None, fps=10.0, out_dir=None):
    mdir = RUNS / model / "models"
    cfg = json.loads((RUNS / model / "config.json").read_text(encoding="utf-8"))
    cols, use, p01 = cfg["features"], cfg["use_model"], cfg["train_p01"]
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    g = video_bins(zones_dir, meta, detections, fps)
    if g.groupby("sid").size().max() < HIST:
        raise SystemExit(f"영상이 짧아요: 예측에는 최근 {HIST * STEP // 60}분이 필요합니다.")
    sci = pd.DataFrame([schedule_row(z, v) for z, v in meta.items()]).set_index("sid", drop=False)
    F, _, _, N0, M = build(g, sci, noise=(0.0, 0.0), future=False)  # 실제 공지는 이미 오차를 담고 있으므로 오차를 더 섞지 않음
    has_sched = M.sid.map({z: bool((v.get("schedule") or {}).get("events")) for z, v in meta.items()}).to_numpy()
    area = F.area.to_numpy()
    # 유입이 학습 범위(하위 1%)보다 작으면 모델이 본 적 없는 상황 -> 직전값으로 대신하고 표시
    in_range = np.all([F[c].to_numpy() >= p01[c] for c in RANGE_COLS], 0)
    rows = {"sec": M.t, "zone": M.sid, "model": np.where(~in_range, "직전값(학습 범위 밖)", np.where(has_sched, "공지", "기준")),
            "in_range": in_range, "n_now": N0, "density_now": N0 / area}
    for h in H:
        k = f"{h // 60}m"
        pb = lgb.Booster(model_file=str(mdir / f"base_{k}.txt")).predict(F[cols["base"]]) if use["base"][k] else np.zeros(len(F))
        ps = lgb.Booster(model_file=str(mdir / f"sched_{k}.txt")).predict(F[cols["sched"]]) if use["sched"][k] else np.zeros(len(F))
        n = np.clip(N0 + np.where(in_range, np.where(has_sched, ps, pb), 0), 0, None)
        rows[f"n_{k}"], rows[f"density_{k}"] = n, n / area
    fc = pd.DataFrame(rows)
    al = []
    for _, r in fc.iterrows():
        for thr, lvl in LEVELS:
            hit = [h // 60 for h in H if r[f"density_{h // 60}m"] >= thr]
            if hit:
                al.append({"sec": r.sec, "zone": r.zone, "level": lvl, "threshold": thr, "first_horizon_min": hit[0],
                           "pred_density": round(r[f"density_{hit[0]}m"], 2), "density_now": round(r.density_now, 2), "model": r.model,
                           "message": f"{r.zone}: {hit[0]}분 뒤 {r[f'density_{hit[0]}m']:.1f}명/㎡ 예상 ({lvl} 기준 {thr:g}명/㎡)"})
                break
    al = pd.DataFrame(al, columns=["sec", "zone", "level", "threshold", "first_horizon_min", "pred_density", "density_now", "model", "message"])
    out = Path(out_dir or zones_dir)
    out.mkdir(parents=True, exist_ok=True)
    fc.to_csv(out / "forecast_minutes.csv", index=False, encoding="utf-8-sig")
    al.to_csv(out / "alarms_minutes.csv", index=False, encoding="utf-8-sig")
    return fc, al


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--out", required=True); t.add_argument("--hide", type=float, default=0.2)
    t.add_argument("--shapes", default="straight,bottleneck")
    t.add_argument("--drop", default="", help="뺄 특징(쉼표). 예: speed,speed_1m,speed_5m — 시뮬레이션과 실제 영상에서 뜻이 다른 특징")
    p = sp.add_parser("predict")
    p.add_argument("--model", required=True); p.add_argument("--zones-dir", required=True); p.add_argument("--meta", required=True)
    p.add_argument("--detections"); p.add_argument("--fps", type=float, default=10.0); p.add_argument("--out-dir")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.out, a.hide, a.shapes.split(","), tuple(c for c in a.drop.split(",") if c))
    else:
        fc, al = predict(a.model, a.zones_dir, a.meta, a.detections, a.fps, a.out_dir)
        print(f"예측 {len(fc)}행, 경보 {len(al)}건")
        print(fc.tail(4).round(2).to_string(index=False))
        if len(al):
            print(al.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
