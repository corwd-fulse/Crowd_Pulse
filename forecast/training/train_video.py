"""쌓인 영상 분석 CSV -> 전처리 -> LightGBM·LSTM 학습 -> Naive와 비교(MAE·RMSE) -> SHAP -> forecast/serving/artifacts/video/.
서비스(run_video_forecast)는 저장된 모델로 학습 없이 바로 예측한다(model=lightgbm_pre, lstm_pre). 서비스 요청 안에서 돌리지 않는 오프라인 작업.

실행(저장소 루트): python -m forecast.training.train_video 결과폴더_또는_상위폴더 ... [--horizons 10 30 60] [--target mean10] [--force]
- 입력: run_zone_pipeline 결과 폴더(crowd_observations.csv, 있으면 flow.csv, 서비스가 남긴 arrival_features.csv·trajectory_features.csv). 상위 폴더를 주면 그 아래를 모두 찾고, 형식이 틀린 폴더는 건너뛴다.
- 전처리·특징: serving.video_pipeline의 load_video_series, _zone_features, pre_design(서비스와 같은 함수). 카메라마다 규모가 달라 비율로 학습.
- 기준선: Naive(지금 값 그대로). 1초 단위 짧은 영상에는 요일·시각 주기가 없어 생활인구의 Seasonal Naive 대신 쓴다.
- 비교·평가: 영상 단위 GroupKFold(처음 보는 영상)로 Naive·LightGBM·LSTM의 MAE·RMSE·skill(= 1 - MAE/MAE(Naive)) -> compare_video.csv.
  같은 영상을 앞뒤로 나누면 결과가 부풀려지므로 쓰지 않는다.
- SHAP: LightGBM 내장 TreeSHAP(pred_contrib)으로 특징별 평균 기여(사람 수) -> manifest와 shap_importance.png.
- 교체: LightGBM은 예측 시간별 skill > 0, LSTM은 평균 skill > 0일 때만 교체(--force 예외). 바뀌는 이전 파일은 history/<이전 버전>/에 보관.
"""
import argparse, copy, json, shutil
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

from ..serving.video_pipeline import (ARRIVAL_CSV, TRAJ_CSV, LGB_WALK_PARAMS, PRE_DIR, PRE_FEATURES, SEQ_FEATURES, SEQ_L, TARGETS, _zone_features,
                                      attach_arrival, attach_traj, load_video_series, pre_design)

MIN_VIDEOS, MIN_ROWS, MAX_FOLDS, SEED = 2, 500, 5, 0
LGB_PARAMS = {**LGB_WALK_PARAMS, "random_state": SEED}  # 'D' 설정(잎 7, 잎당 최소 80행): 광화문 실험에서 기존(잎 31)보다 30·60초 유의하게 나음
LSTM_HIDDEN, LSTM_LR, LSTM_BATCH, LSTM_EPOCHS, LSTM_PATIENCE, LSTM_ROWS_PER_EPOCH = 32, 2e-3, 128, 60, 8, 200_000


def collect(paths):
    """결과 폴더 찾기 + 전처리. 형식이 틀린 폴더는 건너뛰고 기록(하나 때문에 전체 학습이 멈추지 않게)."""
    found = []
    for p in map(Path, paths):
        found += [p] if (p / "crowd_observations.csv").exists() else sorted(q.parent for q in p.rglob("crowd_observations.csv"))
    folders, videos, skipped = [], [], {}
    for d in dict.fromkeys(found):
        try:
            ts = load_video_series(d)[0]
            if (d / ARRIVAL_CSV).exists():  # 서비스가 남긴 도착·이탈 예정(없으면 _zone_features가 0으로 채움)
                ts = attach_arrival(ts, pd.read_csv(d / ARRIVAL_CSV))
            if (d / TRAJ_CSV).exists():  # 서비스가 남긴 역방향·체류(없으면 0)
                ts = attach_traj(ts, pd.read_csv(d / TRAJ_CSV))
        except (ValueError, FileNotFoundError) as e:
            skipped[str(d)] = str(e)[:120]; continue
        folders.append(d); videos.append((ts, _zone_features(ts)))
    return folders, videos, skipped


def build(videos, horizons, target):
    """학습 행. X(표), S(시퀀스), base=y[t], ref, act[행, 예측 시간]=y[t+H], grp(영상 번호)."""
    parts = []
    for vi, (ts, extra) in enumerate(videos):
        for z, g in ts.groupby("zone"):
            d = pre_design(g, extra[z], target)
            y, t = d["y"], d["ts_all"]
            act = np.column_stack([np.where(t + H < len(y), y[np.minimum(t + H, len(y) - 1)], np.nan) for H in horizons]) if len(t) else np.zeros((0, len(horizons)))
            keep = d["ok"] & ~np.isnan(act).all(1)
            parts.append({"X": d["X"][keep], "S": d["S"][keep], "base": y[t][keep], "ref": d["ref"][keep], "act": act[keep], "grp": np.full(int(keep.sum()), vi)})
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    D["Y"] = ((D["act"] - D["base"][:, None]) / D["ref"][:, None]).astype(np.float32)  # 정답: 변화량 비율
    return D


def fit_lgbm(X, y):
    import lightgbm as lgb
    m = ~np.isnan(y)
    return lgb.LGBMRegressor(**LGB_PARAMS).fit(X[m], y[m]).booster_


def fit_lstm(S, Y, S_val=None, Y_val=None, epochs=LSTM_EPOCHS):
    """L1 손실(빈 정답은 제외). 검증 데이터가 있으면 조기 종료. -> (net(cpu), 최적 에폭 수)"""
    import torch
    from ..serving.video_lstm import VideoNet, lstm_predict
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = VideoNet(S.shape[2], LSTM_HIDDEN, Y.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), LSTM_LR)
    best, best_ep, best_state, bad = np.inf, epochs, None, 0
    for ep in range(epochs):
        net.train()
        idx = rng.permutation(len(S))[:LSTM_ROWS_PER_EPOCH]
        for b in range(0, len(idx), LSTM_BATCH):
            j = idx[b:b + LSTM_BATCH]
            x, yv = torch.from_numpy(S[j]).to(dev), torch.from_numpy(Y[j]).to(dev)
            m = ~torch.isnan(yv)
            loss = (net(x) - torch.nan_to_num(yv)).abs()[m].mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if S_val is not None:
            mv = ~np.isnan(Y_val)
            v = float(np.abs(lstm_predict(net, S_val) - np.nan_to_num(Y_val))[mv].mean())
            if v < best - 1e-6:
                best, best_ep, best_state, bad = v, ep + 1, copy.deepcopy(net.state_dict()), 0
            elif (bad := bad + 1) >= LSTM_PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net.cpu().eval(), best_ep


def _holdout(idx, grp):
    """조기 종료용으로 학습 행 중 영상 1개를 떼어낸다(영상이 1개뿐이면 None)."""
    gs = np.unique(grp[idx])
    if len(gs) < 2:
        return idx, None
    return idx[grp[idx] != gs[-1]], idx[grp[idx] == gs[-1]]


def cross_validate(D, horizons):
    """영상 단위 교차검증으로 Naive·LightGBM·LSTM 비교 -> DataFrame[horizon_s, model, MAE, RMSE, skill, n]."""
    from sklearn.model_selection import GroupKFold
    from ..serving.video_lstm import lstm_predict
    X, S, Y, grp = D["X"], D["S"], D["Y"], D["grp"]
    PL, PS = np.full(Y.shape, np.nan), np.full(Y.shape, np.nan)
    for tr, te in GroupKFold(min(MAX_FOLDS, len(np.unique(grp)))).split(X, groups=grp):
        for j in range(len(horizons)):
            PL[te, j] = fit_lgbm(X[tr], Y[tr, j]).predict(X[te])
        fit_i, val_i = _holdout(tr, grp)
        net, _ = fit_lstm(S[fit_i], Y[fit_i], *((S[val_i], Y[val_i]) if val_i is not None else ()))
        PS[te] = lstm_predict(net, S[te])
    rows = []
    for j, H in enumerate(horizons):
        a, ok = D["act"][:, j], ~np.isnan(D["act"][:, j])
        e0 = np.abs(D["base"] - a)[ok].mean()
        for name, pr in (("naive", D["base"]), ("lightgbm_pre", D["base"] + PL[:, j] * D["ref"]), ("lstm_pre", D["base"] + PS[:, j] * D["ref"])):
            err = (pr - a)[ok]
            mae = float(np.abs(err).mean())
            rows.append({"horizon_s": H, "model": name, "MAE": round(mae, 4), "RMSE": round(float(np.sqrt((err ** 2).mean())), 4),
                         "skill": round(1 - mae / e0, 4) if e0 > 0 else None, "n": int(ok.sum())})
    return pd.DataFrame(rows)


def shap_importance(booster, X, ref, n=20_000):
    """특징별 평균 |SHAP| (사람 수 단위). LightGBM 내장 TreeSHAP."""
    i = np.random.default_rng(SEED).permutation(len(X))[:n]
    c = booster.predict(X[i], pred_contrib=True)[:, :-1] * ref[i, None]
    return dict(sorted(((f, round(float(v), 4)) for f, v in zip(PRE_FEATURES, np.abs(c).mean(0))), key=lambda kv: -kv[1]))


def _plot_shap(imp, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(imp), figsize=(4.5 * len(imp), 4), squeeze=False)
    for ax, (key, d) in zip(axes[0], imp.items()):
        f = list(d)[::-1]
        ax.barh(f, [d[k] for k in f], color="#4C78A8")
        ax.set_title(f"LightGBM {key}: mean |SHAP| (persons)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _replace(out, fname, manifest_key, manifest):
    """바뀌는 이전 파일을 history/<이전 버전>/으로 옮긴다."""
    old = manifest["models"].get(manifest_key)
    if old and (out / fname).exists():
        (out / "history" / old["version"]).mkdir(parents=True, exist_ok=True)
        shutil.move(str(out / fname), str(out / "history" / old["version"] / fname))
    return old["version"] if old else None


def train(paths, horizons=(10, 30, 60), target="mean10", out_dir=None, force=False):
    import torch
    if target not in TARGETS:
        raise ValueError(f"target은 {TARGETS} 중 하나")
    horizons = tuple(sorted(set(int(h) for h in horizons)))
    folders, videos, skipped = collect(paths)
    if len(folders) < MIN_VIDEOS:
        raise ValueError(f"영상 {len(folders)}개: 영상 단위 검증을 하려면 최소 {MIN_VIDEOS}개가 필요합니다. 건너뛴 폴더: {skipped}")
    D = build(videos, horizons, target)
    if len(D["Y"]) < MIN_ROWS:
        raise ValueError(f"학습 행 {len(D['Y'])}개 < {MIN_ROWS}: 영상이 더 쌓인 뒤 학습하세요.")
    comp = cross_validate(D, horizons)
    skill = {(r.model, r.horizon_s): r.skill for r in comp.itertuples()}
    out = Path(out_dir or PRE_DIR); out.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out / "compare_video.csv", index=False, encoding="utf-8-sig")
    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {"models": {}}
    version, now = "video-" + datetime.now().strftime("%Y%m%d-%H%M%S"), datetime.now().isoformat(timespec="seconds")
    common = {"version": version, "target": target, "trained_at": now, "train_videos": [str(f) for f in folders], "rows": int(len(D["Y"]))}
    saved, imp = {}, {}
    for j, H in enumerate(horizons):
        key, fname, s = f"lgbm_{target}_{H}s", f"lgbm_{target}_{H}s.txt", skill[("lightgbm_pre", H)]
        if not (force or (s is not None and s > 0)):
            saved[key] = f"유지(처음 보는 영상 skill {s} <= 0)"; continue
        b = fit_lgbm(D["X"], D["Y"][:, j])
        imp[f"{H}s"] = shap_importance(b, D["X"], D["ref"])
        prev = _replace(out, fname, key, manifest)
        b.save_model(str(out / fname))
        manifest["models"][key] = {**common, "file": fname, "horizon_s": H, "features": PRE_FEATURES, "skill": s,
                                   "shap_importance": imp[f"{H}s"], "previous_version": prev}
        saved[key] = "교체"
    key, fname = f"lstm_{target}", f"lstm_{target}.pt"
    s_lstm = float(np.mean([skill[("lstm_pre", H)] for H in horizons]))
    if force or s_lstm > 0:
        fit_i, val_i = _holdout(np.arange(len(D["Y"])), D["grp"])
        best_ep = fit_lstm(D["S"][fit_i], D["Y"][fit_i], D["S"][val_i], D["Y"][val_i])[1] if val_i is not None else LSTM_EPOCHS  # 에폭 수 정하기
        net, _ = fit_lstm(D["S"], D["Y"], epochs=best_ep)                                # 전체로 다시 학습
        prev = _replace(out, fname, key, manifest)
        torch.save({"state": net.state_dict(), "n_in": len(SEQ_FEATURES), "hidden": LSTM_HIDDEN, "horizons": list(horizons),
                    "seq_len": SEQ_L, "target": target}, out / fname)
        manifest["models"][key] = {**common, "file": fname, "horizons": list(horizons), "features": SEQ_FEATURES, "seq_len": SEQ_L,
                                   "epochs": best_ep, "skill_mean": round(s_lstm, 4), "previous_version": prev}
        saved[key] = "교체"
    else:
        saved[key] = f"유지(처음 보는 영상 평균 skill {s_lstm:.3f} <= 0)"
    if imp:
        _plot_shap(imp, out / "shap_importance.png")
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"version": version, "videos": len(folders), "rows": int(len(D["Y"])), "compare": comp, "saved": saved, "skipped_folders": skipped}


def main():
    ap = argparse.ArgumentParser(description="쌓인 영상 분석 CSV로 구역 인원 예측 LightGBM·LSTM을 미리 학습한다(오프라인).")
    ap.add_argument("paths", nargs="+", help="결과 폴더(crowd_observations.csv가 있는 폴더) 또는 그 상위 폴더")
    ap.add_argument("--horizons", type=int, nargs="+", default=[10, 30, 60])
    ap.add_argument("--target", default="mean10", choices=TARGETS)
    ap.add_argument("--out-dir", help=f"저장 위치(기본 {PRE_DIR})")
    ap.add_argument("--force", action="store_true", help="검증 결과와 상관없이 교체")
    a = ap.parse_args()
    r = train(a.paths, a.horizons, a.target, a.out_dir, a.force)
    print(f"영상 {r['videos']}개, 학습 행 {r['rows']:,}, 버전 {r['version']}")
    print(r["compare"].to_string(index=False))
    print("저장:", r["saved"])
    if r["skipped_folders"]:
        print("건너뛴 폴더:", r["skipped_folders"])


if __name__ == "__main__":
    main()
