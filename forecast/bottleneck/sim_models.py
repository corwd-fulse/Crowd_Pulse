"""JuPedSim 시나리오(sim_dataset)로 5/10/15분 뒤 구역 인원 예측: 직전값 / LightGBM / LSTM(시드 3) / 앙상블 비교 표.

- 10초 칸으로 묶고, 예측 시점(30초마다)까지 최근 15분만 입력. 대상 = h초 뒤 인원 - 지금 인원.
- 분할: 시나리오 단위 학습 70% / 검증 15% / 테스트 15% (테스트 = 처음 보는 폭·출구 폭 조합).
- 공간 특성(형태, 통로 폭, 출구 폭, 면적, 출구 용량, 유입/용량 비)을 넣은 것과 뺀 것을 같이 학습해 효과 확인.
- 앙상블: 평균, 가중(검증 MAE가 가장 작은 LightGBM 몫 w, 0~1 0.1 간격).
- 차이 신뢰구간: 테스트 시나리오 단위 부트스트랩 2,000회.
- --schedule: 예정된 몰림(공지된 일정) 특징 추가. 시나리오 몰림 계획에 오차(시각 ±3분, 크기 ±30%, 시나리오마다 고정)를 섞어
  앞으로 5/10/15분 예상 유입, 다음 몰림까지 남은 시간·크기를 넣는다. 현실의 공연 종료·열차 도착 시각에 해당.
실행(data-area 루트): C:/Users/Woori/venvs/nf/Scripts/python.exe -m forecast.bottleneck.sim_models --out 20261120_180_schedule --schedule
출력: forecast/bottleneck/runs/<out>/ (config.json, results.md, results_*.csv, split.csv, test_predictions.parquet) — 실행마다 폴더 하나"""
import argparse, json, subprocess, time, zlib
from pathlib import Path

import numpy as np, pandas as pd
import lightgbm as lgb
import torch, torch.nn as nn
from scipy.special import ndtr

from .sim_dataset import OUT, CAP

RUNS = Path(__file__).resolve().parent / "runs"

H = (300, 600, 900)
STEP, HIST, EVERY = 10, 90, 3  # 10초 칸, 입력 15분, 예측 시점 30초마다
SEQ = ["density", "in_w", "out_w", "speed", "heading", "in_r_w"]
SPATIAL = ["is_bottleneck", "is_tjunction", "width", "exit_width", "entry_width", "area", "cap", "in_cap"]
SCHED = [f"plan_in_{h // 60}m" for h in H] + ["plan_next_in", "plan_next_h"]
SCHED_DT, SCHED_DH = 180.0, 0.3  # 공지 일정 오차: 시각 ±3분, 크기 ±30% (균등)


def noisy_plan(row, dt=SCHED_DT, dh=SCHED_DH, hide=0.0):
    """시나리오 몰림 계획 + 오차 -> (평소 유입 합, [(중심, 폭, 높이)]). 오차는 시나리오마다 고정(같은 공지를 계속 본다).
    hide: 몰림마다 이 확률로 '공지 안 됨'(목록에서 뺌). 1이면 공지 없음(평소 유입만)."""
    sid = row["sid"]
    seed = int(sid) if isinstance(sid, (int, float, np.integer)) else zlib.crc32(str(sid).encode())  # 서비스: 구역 이름
    rng = np.random.default_rng([1, seed])
    rng_h = np.random.default_rng([2, seed])  # 숨김은 따로 뽑아 오차 값이 hide와 무관하게 같게
    base, bumps = 0.0, []
    for s in ("", "_r"):
        if isinstance(row.get("bumps" + s), str):
            base += row["base" + s]
            bumps += [(c + rng.uniform(-dt, dt), w, h * rng.uniform(1 - dh, 1 + dh)) for c, w, h in json.loads(row["bumps" + s])]
    keep = rng_h.uniform(size=len(bumps)) >= hide
    return base, [b for b, k in zip(bumps, keep) if k]


def schedule_features(row, t, cap, dt=SCHED_DT, dh=SCHED_DH, hide=0.0):
    """예측 시점 t(초 배열)마다 공지 일정 특징. 예상 유입 = 계획 유입률을 (t, t+h] 적분(가우시안은 정규 누적분포로).
    shortcut: 입구 폭 한계(sim_dataset.rate의 1.8명/m/s)는 빼고 적분, 한계에 자주 닿으면 수치 적분으로 바꿀 것."""
    base, bumps = noisy_plan(row, dt, dh, hide)
    t = np.asarray(t, float)
    out = {}
    for h in H:
        v = np.full(len(t), base * h) + sum(hh * w * np.sqrt(2 * np.pi) * (ndtr((t + h - c) / w) - ndtr((t - c) / w)) for c, w, hh in bumps)
        out[f"plan_in_{h // 60}m"] = v
    nxt = np.full(len(t), 3600.0); nh = np.zeros(len(t))
    for c, w, hh in sorted(bumps):
        m = (c > t) & (nxt == 3600.0)  # 아직 안 온 첫 몰림(정점 기준)
        nxt[m], nh[m] = c - t[m], hh / cap
    out["plan_next_in"], out["plan_next_h"] = nxt, nh
    return out


def load():
    sc = pd.read_csv(OUT / "scenarios.csv")
    parts = sorted((OUT / "parts").glob("*.parquet"))
    s = pd.concat([pd.read_parquet(f) for f in parts])
    if "in_right" not in s:  # T자 추가 전에 만든 파일: 한 방향 유입
        s["in_right"] = np.nan
    s["in_right"] = s.in_right.fillna(0)
    s["b"] = s.t // STEP
    g = s.groupby(["sid", "b"]).agg(n=("n", "last"), inflow=("inflow", "sum"), outflow=("outflow", "sum"), in_right=("in_right", "sum"),
                                    speed=("speed", "mean"), heading=("heading", "mean")).reset_index()
    if "branch_width" not in sc:
        sc["branch_width"] = np.nan
    sc["entry_width"] = sc.branch_width.fillna(sc.width)
    g = g.merge(sc[["sid", "shape", "width", "exit_width", "entry_width", "area"]], on="sid")
    g["density"] = g.n / g.area
    g["in_w"], g["out_w"], g["in_r_w"] = g.inflow / g.width, g.outflow / g.width, g.in_right / g.width  # 폭 1m당 10초 유입·유출
    g[["speed", "heading"]] = g[["speed", "heading"]].fillna(0)  # 빈 구역
    return g, sc[sc.sid.isin(g.sid.unique())]


def build(g, sc=None, noise=(SCHED_DT, SCHED_DH), hide=0.0, future=True):
    """예측 시점마다 LightGBM 특징, LSTM 시퀀스, 정답. sc를 주면 공지 일정 특징(SCHED)도 넣는다.
    future=False(서비스): 정답이 없는 최신 시점까지 매 칸 만들고, 모르는 정답은 nan."""
    F, X, Y, N0, meta = [], [], [], [], []
    hk = [h // STEP for h in H]
    every, tail = (EVERY, max(hk)) if future else (1, 0)
    for sid, d in g.groupby("sid"):
        d = d.sort_values("b")
        n, a = d.n.to_numpy(float), d.area.iloc[0]
        inf, outf, sp, inr = d.inflow.to_numpy(float), d.outflow.to_numpy(float), d.speed.to_numpy(float), d.in_right.to_numpy(float)
        seq = d[SEQ].to_numpy(np.float32)
        cap = CAP * d.exit_width.iloc[0]
        if sc is not None:
            idx = list(range(HIST - 1, len(d) - tail, every))
            plan = schedule_features(sc.loc[sid], d.b.to_numpy()[idx] * STEP + STEP, cap, *noise, hide)
        for j, i in enumerate(range(HIST - 1, len(d) - tail, every)):
            w = lambda x, k: x[i - k + 1:i + 1].sum()
            f = {"n": n[i], "density": n[i] / a, "speed": sp[i], "speed_1m": sp[i - 5:i + 1].mean(), "speed_5m": sp[i - 29:i + 1].mean(),
                 "heading": d.heading.iloc[i]}
            for k, nm in ((6, "1m"), (30, "5m"), (90, "15m")):
                f[f"in_{nm}"], f[f"out_{nm}"] = w(inf, k), w(outf, k)
                f[f"dn_{nm}"] = n[i] - n[i - k + 1] if k < HIST else n[i] - n[i - HIST + 1]
            f["in_trend"] = w(inf, 30) - inf[i - 59:i - 29].sum()  # 최근 5분 유입 - 그 전 5분
            f["in_right_share_5m"] = w(inr, 30) / max(w(inf, 30), 1)  # 이동 방향: 오른쪽에서 온 비율(한 방향 형태는 0)
            shp = d["shape"].iloc[0]
            f.update(is_bottleneck=float(shp == "bottleneck"), is_tjunction=float(shp == "tjunction"), width=d.width.iloc[0],
                     exit_width=d.exit_width.iloc[0], entry_width=d.entry_width.iloc[0], area=a, cap=cap, in_cap=f["in_5m"] / 300 / cap)
            if sc is not None:
                f.update({k: v[j] for k, v in plan.items()})
            F.append(f); X.append(seq[i - HIST + 1:i + 1]); N0.append(n[i])
            Y.append([n[i + k] - n[i] if i + k < len(n) else np.nan for k in hk]); meta.append((sid, d["shape"].iloc[0], int(d.b.iloc[i]) * STEP + STEP))
    m = pd.DataFrame(meta, columns=["sid", "shape", "t"])  # t = 예측 시점(초, 그 칸 끝)
    return pd.DataFrame(F), np.stack(X), np.array(Y), np.array(N0), m


def fit_lgbm(Ftr, ytr, Fva, yva, Fte, cols, return_models=False):
    p = dict(objective="l1", learning_rate=0.03, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.8, verbose=-1, seed=0, num_threads=2)  # 시뮬레이션과 같이 돌 때 스레드 다툼 방지
    out_va, out_te, models = [], [], []
    for k in range(len(H)):
        m = lgb.train(p, lgb.Dataset(Ftr[cols], ytr[:, k]), 2000, valid_sets=[lgb.Dataset(Fva[cols], yva[:, k])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        out_va.append(m.predict(Fva[cols], num_iteration=m.best_iteration)); out_te.append(m.predict(Fte[cols], num_iteration=m.best_iteration))
        models.append(m)
    return (np.stack(out_va, 1), np.stack(out_te, 1)) + ((models,) if return_models else ())


class Net(nn.Module):
    def __init__(s, n_static, h=64):
        super().__init__()
        s.lstm = nn.LSTM(len(SEQ), h, num_layers=2, batch_first=True, dropout=0.1)
        s.head = nn.Sequential(nn.Linear(h + n_static, 64), nn.ReLU(), nn.Linear(64, len(H)))

    def forward(s, x, st):
        o, _ = s.lstm(x)
        return s.head(torch.cat([o[:, -1], st], -1))


def fit_lstm(Xtr, Str, ytr, Xva, Sva, yva, Xte, Ste, seed, epochs=60, patience=8):
    """대상 = 밀도 변화(인원 변화 / 면적). 시퀀스·정적 입력은 학습 평균·표준편차로 표준화."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = Xtr.reshape(-1, Xtr.shape[2]).mean(0), Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-6
    smu, ssd = Str.mean(0), Str.std(0) + 1e-6
    T = lambda X, S: (torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev), torch.tensor((S - smu) / ssd, dtype=torch.float32, device=dev))
    tr, va, te = T(Xtr, Str), T(Xva, Sva), T(Xte, Ste)
    yt, yv = torch.tensor(ytr, dtype=torch.float32, device=dev), torch.tensor(yva, dtype=torch.float32, device=dev)
    net = Net(Str.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    pred = lambda d: torch.cat([net(d[0][i:i + 2048], d[1][i:i + 2048]) for i in range(0, len(d[0]), 2048)])
    best, state, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        for b in torch.randperm(len(yt), device=dev).split(256):
            loss = (net(tr[0][b], tr[1][b]) - yt[b]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = (pred(va) - yv).abs().mean().item()
        if v < best:
            best, state, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    with torch.no_grad():
        return pred(va).cpu().numpy(), pred(te).cpu().numpy(), ep + 1


def md_table(d):
    rows = [list(map(str, d.columns))] + d.astype(str).values.tolist()
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def boot_ci(err_a, err_b, sids, n=2000, seed=0):
    """시나리오 단위 부트스트랩: MAE(a) - MAE(b)의 95% 구간."""
    rng = np.random.default_rng(seed)
    u = np.unique(sids); idx = [np.where(sids == s)[0] for s in u]
    diffs = []
    for _ in range(n):
        pick = np.concatenate([idx[j] for j in rng.integers(0, len(u), len(u))])
        diffs.append(err_a[pick].mean() - err_b[pick].mean())
    return np.percentile(diffs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", required=True, help="실행 이름 = runs/ 아래 폴더 이름(예: 20261120_180_schedule)")
    ap.add_argument("--schedule", action="store_true", help="예정된 몰림(공지 일정) 특징 추가")
    ap.add_argument("--sched-dt", type=float, default=SCHED_DT, help="공지 시각 오차(초, ±균등)")
    ap.add_argument("--sched-dh", type=float, default=SCHED_DH, help="공지 크기 오차(비율, ±균등)")
    ap.add_argument("--shapes", help="쓸 형태만(쉼표): straight,bottleneck,tjunction. 없으면 전부")
    a = ap.parse_args()
    t0 = time.time()
    RUN = RUNS / a.out
    if RUN.exists() and any(RUN.iterdir()):
        raise SystemExit(f"{RUN} 가 이미 있어요. 다른 --out 이름을 쓰세요(이전 결과 덮어쓰기 방지).")
    RUN.mkdir(parents=True, exist_ok=True)
    g, sc = load()
    if a.shapes:
        sc = sc[sc["shape"].isin(a.shapes.split(","))]; g = g[g.sid.isin(sc.sid)]
    F, X, Y, N0, M = build(g, sc.set_index("sid", drop=False) if a.schedule else None, (a.sched_dt, a.sched_dh))
    rng = np.random.default_rng(0)
    sids = rng.permutation(sc.sid.to_numpy())
    ntr, nva = int(len(sids) * 0.7), int(len(sids) * 0.15)
    part = {s: "train" for s in sids[:ntr]} | {s: "val" for s in sids[ntr:ntr + nva]} | {s: "test" for s in sids[ntr + nva:]}
    M["part"] = M.sid.map(part)
    tr, va, te = [(M.part == p).to_numpy() for p in ("train", "val", "test")]
    print(f"시나리오 {len(sids)}개(학습 {ntr}, 검증 {nva}, 테스트 {len(sids) - ntr - nva}), 예측 시점 학습 {tr.sum():,} 검증 {va.sum():,} 테스트 {te.sum():,}")
    area = F.area.to_numpy()
    Yte = Y[te]
    nonsp = [c for c in F.columns if c not in SPATIAL]

    preds = {"직전값": np.zeros_like(Yte)}
    lg_va, lg_te = fit_lgbm(F[tr], Y[tr], F[va], Y[va], F[te], list(F.columns))
    lgn_va, lgn_te = fit_lgbm(F[tr], Y[tr], F[va], Y[va], F[te], nonsp)
    preds["LightGBM"], preds["LightGBM (공간 특성 뺌)"] = lg_te, lgn_te
    print(f"LightGBM 끝 {time.time() - t0:.0f}s", flush=True)

    S = F[SPATIAL].to_numpy(np.float32)
    if a.schedule:  # 공지 일정은 시점마다 바뀌는 정적 입력으로 LSTM 머리에 붙인다
        S = np.concatenate([S, F[SCHED].to_numpy(np.float32)], 1)
    S0 = F[SCHED].to_numpy(np.float32) if a.schedule else np.zeros((len(F), 1), np.float32)  # 공간 특성 없는 LSTM
    Yd = Y / area[:, None]  # 밀도 변화
    lstm_va, lstm_te, lstmn_te, eps = [], [], [], []
    for seed in range(a.seeds):
        pv, pt, ep = fit_lstm(X[tr], S[tr], Yd[tr], X[va], S[va], Yd[va], X[te], S[te], seed)
        _, ptn, _ = fit_lstm(X[tr], S0[tr], Yd[tr], X[va], S0[va], Yd[va], X[te], S0[te], seed)
        lstm_va.append(pv * area[va, None]); lstm_te.append(pt * area[te, None]); lstmn_te.append(ptn * area[te, None]); eps.append(ep)
        print(f"LSTM 시드 {seed} {ep}에폭, {time.time() - t0:.0f}s", flush=True)
    lstm_va_m, lstm_te_m = np.mean(lstm_va, 0), np.mean(lstm_te, 0)

    # 가중 앙상블: 시간 구간마다 검증 MAE가 가장 작은 LightGBM 몫
    ws = np.round(np.arange(0, 1.01, 0.1), 1)
    wbest = [ws[np.argmin([np.abs(w * lg_va[:, k] + (1 - w) * lstm_va_m[:, k] - Y[va][:, k]).mean() for w in ws])] for k in range(len(H))]
    preds["LSTM (시드 평균 예측)"] = lstm_te_m
    preds["LSTM (공간 특성 뺌)"] = np.mean(lstmn_te, 0)
    preds["앙상블 평균"] = (lg_te + lstm_te_m) / 2
    preds["앙상블 가중"] = np.stack([wbest[k] * lg_te[:, k] + (1 - wbest[k]) * lstm_te_m[:, k] for k in range(len(H))], 1)

    E = {k: np.abs(p - Yte) for k, p in preds.items()}
    base = E["직전값"].mean(0)
    seed_mae = np.array([np.abs(p - Yte).mean(0) for p in lstm_te])  # 시드별 MAE
    rows = []
    for name in ["직전값", "LightGBM", "LSTM (시드 평균 예측)", "앙상블 평균", "앙상블 가중"]:
        r = {"모델": name}
        for k, h in enumerate(H):
            mae = E[name][:, k].mean()
            r[f"MAE {h // 60}분"] = f"{mae:.2f}" + (f" (시드별 {seed_mae[:, k].mean():.2f}±{seed_mae[:, k].std():.2f})" if name.startswith("LSTM") else "")
            r[f"개선율 {h // 60}분"] = f"{1 - mae / base[k]:.1%}"
        rows.append(r)
    main_t = pd.DataFrame(rows)

    sid_te = M.sid[te].to_numpy()
    ci_rows = []
    for a_, b_ in (("앙상블 가중", "LightGBM"), ("앙상블 가중", "LSTM (시드 평균 예측)"), ("LightGBM", "LSTM (시드 평균 예측)")):
        r = {"비교 (앞 - 뒤 MAE)": f"{a_} - {b_}"}
        for k, h in enumerate(H):
            lo, hi = boot_ci(E[a_][:, k], E[b_][:, k], sid_te)
            r[f"{h // 60}분 95% 구간"] = f"[{lo:+.2f}, {hi:+.2f}]" + (" 유의" if hi < 0 or lo > 0 else "")
        ci_rows.append(r)
    ci_t = pd.DataFrame(ci_rows)

    sp_rows = []
    for m_with, m_wo in (("LightGBM", "LightGBM (공간 특성 뺌)"), ("LSTM (시드 평균 예측)", "LSTM (공간 특성 뺌)")):
        r = {"모델": m_with.split(" ")[0]}
        for k, h in enumerate(H):
            lo, hi = boot_ci(E[m_with][:, k], E[m_wo][:, k], sid_te)
            r[f"{h // 60}분 넣음/뺌"] = f"{E[m_with][:, k].mean():.2f} / {E[m_wo][:, k].mean():.2f}" + (" 유의" if hi < 0 or lo > 0 else "")
        sp_rows.append(r)
    sp_t = pd.DataFrame(sp_rows)

    sh_rows = []
    for shp in ("straight", "bottleneck", "tjunction"):
        mk = (M["shape"][te] == shp).to_numpy()
        if not mk.any():
            continue
        for name in ["직전값", "LightGBM", "LSTM (시드 평균 예측)", "앙상블 가중"]:
            sh_rows.append({"형태": shp, "모델": name, **{f"MAE {h // 60}분": round(E[name][mk, k].mean(), 2) for k, h in enumerate(H)},
                            "평균 인원": round(N0[te][mk].mean(), 1)})
    sh_t = pd.DataFrame(sh_rows)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=RUNS.parent).stdout.strip()
    (RUN / "config.json").write_text(json.dumps(dict(out=a.out, schedule=a.schedule, seeds=a.seeds, horizons_s=H, step_s=STEP, hist_bins=HIST,
                                                     sched_noise=dict(time_s=a.sched_dt, size=a.sched_dh) if a.schedule else None, shapes_arg=a.shapes,
                                                     n_scenarios=len(sids), shapes=sc["shape"].value_counts().to_dict(), features=list(F.columns),
                                                     git_commit=commit, created=time.strftime("%Y-%m-%d %H:%M")), ensure_ascii=False, indent=1), encoding="utf-8")
    # 뷰어용 테스트 예측(인원 = 지금 인원 + 예측 변화)
    mt = M[te].reset_index(drop=True)
    long = [mt.assign(horizon=h, n0=N0[te], actual=N0[te] + Yte[:, k], **{nm: N0[te] + p[:, k] for nm, p in preds.items() if "뺌" not in nm})
            for k, h in enumerate(H)]
    pd.concat(long).to_parquet(RUN / "test_predictions.parquet", index=False)
    M[["sid", "part"]].drop_duplicates().to_csv(RUN / "split.csv", index=False)
    main_t.to_csv(RUN / "results_main.csv", index=False, encoding="utf-8-sig")
    sp_t.to_csv(RUN / "results_spatial.csv", index=False, encoding="utf-8-sig")
    sh_t.to_csv(RUN / "results_shape.csv", index=False, encoding="utf-8-sig")
    info = (f"시나리오 {len(sids)}개 (학습 {ntr}/검증 {nva}/테스트 {len(sids) - ntr - nva}), 테스트 예측 시점 {te.sum():,}, "
            f"테스트 평균 인원 {N0[te].mean():.1f}명, LSTM 에폭(조기 종료) {eps}, 가중 앙상블 LightGBM 몫 {dict(zip([f'{h // 60}분' for h in H], map(float, wbest)))}")
    md = "\n\n".join([f"# 시뮬레이션 5/10/15분 예측 비교 ({a.out}{', 공지 일정 특징 있음' if a.schedule else ''})\n\n{info}",
                      "## 모델 비교 (MAE 단위: 명)\n\n" + md_table(main_t),
                      "## 차이 신뢰구간 (시나리오 부트스트랩)\n\n" + md_table(ci_t),
                      "## 공간 특성 효과 (MAE 넣음 / 뺌)\n\n" + md_table(sp_t),
                      "## 형태별\n\n" + md_table(sh_t)])
    (RUN / "results.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n전체 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
