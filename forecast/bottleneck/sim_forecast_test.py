"""예측 → 시뮬레이션 연결 테스트(FR-SIM-02): 시각 t에 실제로 있는 사람을 실제 위치에 놓고, 앞으로 들어올 사람(유입)을 넣어
GA로 보정한 JuPedSim을 돌려 10·20·30초 뒤 문 앞 밀도와 병목(3명/㎡ 이상 시간)을 예측한다.
유입 시나리오: oracle(실제 유입 그대로 = 유입 예측이 완벽할 때의 상한) / rate(최근 10초 유입 속도가 이어진다고 가정 = 단순 유입 예측) / none(유입 없음).
비교: 직전값, 어제 LightGBM·대기열 모형(10초 뒤, 회차 하나씩 빼고 학습, pred_bottleneck.csv), 기본 파라미터 시뮬레이션.
평가: GA 보정에 안 쓴 검증 6회차, t = 3초부터 5초 간격.
실행(data-area 루트): C:/Users/Woori/venvs/nf/Scripts/python.exe -m forecast.bottleneck.sim_forecast_test
출력: data/crowdqueue/simfc_rows.csv, simfc_summary.csv, simfc.png"""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

from .calibrate_ga import DATA, FPS, CAUTION, GENES, DEFAULT, load_h0_runs, real_metrics, simulate, seconds_metrics

HS, STEP, WIN, RATE_WIN = (10, 20, 30), 5, 30, 10
_R = {}


def _init():
    _R.update({r["name"]: r for r in load_h0_runs()})


def scenario_traj(run, t, kind, seed=0):
    """시각 t의 상태 + 유입 시나리오 -> simulate가 읽는 궤적 표(id, fr, x, y). fr 0 = 시각 t."""
    d = run["traj"]; f0 = t * FPS
    first = d.groupby("id").fr.min()
    now = d[(d.fr == f0) & (d.y > -0.3)]  # 지금 보이고 아직 문을 지나지 않은 사람
    rows = [(pid, 0, x, y) for pid, x, y in zip(now.id, now.x, now.y)]
    if kind == "oracle":
        fut = d[d.id.isin(first[first > f0].index)].sort_values("fr").groupby("id").first()
        rows += [(pid, int(r.fr - f0), r.x, r.y) for pid, r in fut.iterrows()]
    elif kind == "rate":
        n_in = int(((first > max(f0 - RATE_WIN * FPS, 0)) & (first <= f0)).sum())
        k = int(round(n_in / RATE_WIN * WIN))
        rng = np.random.default_rng(seed); w = run["w"]
        ys = d[d.fr == d.fr.min()].y
        y0 = min(6.4, float(d.groupby("id").y.first().max()))  # 사람이 들어오던 위쪽 끝
        rows += [(10_000 + i, int((i + 0.5) * WIN / max(k, 1) * FPS), rng.uniform(-w / 2 + 0.3, w / 2 - 0.3), y0) for i in range(k)]
    return pd.DataFrame(rows, columns=["id", "fr", "x", "y"]).assign(z=1.7)


def job(args):
    name, t, kind, plab, g = args
    run = _R[name]
    tr = scenario_traj(run, t, kind)
    if tr.empty:
        return name, t, kind, plab, np.zeros(WIN + 1)
    pos, n = simulate({**run, "traj": tr}, np.asarray(g))
    dens, _ = seconds_metrics(pos, run["w"], max(n, WIN + 1))
    return name, t, kind, plab, dens[: WIN + 1]


def main():
    meta = json.loads((DATA / "ga_best_v2.json").read_text(encoding="utf-8"))
    g2 = [meta["best"].get(k, GENES[k][2]) for k in GENES]
    runs = {r["name"]: r for r in load_h0_runs() if r["name"] in meta["test"]}
    jobs = []
    for name, r in runs.items():
        T = int(r["traj"].fr.max() // FPS) + 1
        for t in range(3, max(T - 10, 4), STEP):
            jobs += [(name, t, k, "GA v2", g2) for k in ("oracle", "rate", "none")] + [(name, t, "oracle", "기본값", list(DEFAULT))]
    print(f"시뮬레이션 {len(jobs)}개", flush=True)
    with ProcessPoolExecutor(7, initializer=_init) as ex:
        out = list(ex.map(job, jobs))
    lgb = pd.read_csv(DATA / "pred_bottleneck.csv")
    lgb = lgb[lgb.h == 10].set_index(["run", "sec"])
    rows = []
    for name, t, kind, plab, dens in out:
        r = runs[name]; rd, _ = real_metrics(r); T = int(r["traj"].fr.max() // FPS) + 1
        for h in HS:
            if t + h >= T:
                continue
            row = {"run": name, "w": r["w"], "t": t, "h": h, "method": f"시뮬({plab}, 유입 {kind})", "pred": dens[h], "real": rd[t + h], "pers": rd[t],
                   "over3_pred": int((dens[1:WIN + 1] >= CAUTION).sum()), "over3_real": int((rd[t + 1:t + WIN + 1] >= CAUTION).sum())}
            rows.append(row)
    R = pd.DataFrame(rows)
    base = R.drop_duplicates(["run", "t", "h"]).copy()
    pers = base.assign(method="직전값", pred=base.pers, over3_pred=np.nan)
    k10 = base[base.h == 10].copy()
    key = list(zip(k10.run, k10.t))
    have = [kk in lgb.index for kk in key]
    k10 = k10[have]
    lg = k10.assign(method="LightGBM(어제, 10초)", pred=[lgb.loc[kk, "lgb"] for kk in zip(k10.run, k10.t)], over3_pred=np.nan)
    qu = k10.assign(method="대기열 모형(어제, 10초)", pred=[lgb.loc[kk, "queue"] for kk in zip(k10.run, k10.t)], over3_pred=np.nan)
    A = pd.concat([R, pers, lg, qu], ignore_index=True)
    A["err"] = (A.pred - A.real).abs()
    A["hit3"] = (A.pred >= CAUTION) == (A.real >= CAUTION)
    A.round(3).to_csv(DATA / "simfc_rows.csv", index=False, encoding="utf-8-sig")
    # 같은 (회차, t, h) 행끼리 비교
    common = A.groupby(["run", "t", "h"]).method.transform("count")
    S = A.groupby(["h", "method"]).agg(MAE=("err", "mean"), 주의판정_일치=("hit3", "mean"), n=("err", "size")).reset_index()
    O = A.dropna(subset=["over3_pred"]).drop_duplicates(["run", "t", "method"]).assign(oe=lambda x: (x.over3_pred - x.over3_real).abs())
    S2 = O.groupby("method").agg(향후30초_3명이상_초_오차=("oe", "mean"), n=("oe", "size")).reset_index()
    S.round(3).to_csv(DATA / "simfc_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(S.round(3).to_string(index=False)); print(S2.round(2).to_string(index=False))
    # 같은 10초 행만으로 LightGBM·대기열과 비교
    m10 = A[(A.h == 10) & A.set_index(["run", "t"]).index.isin(list(zip(k10.run, k10.t)))]
    print("10초 뒤, LightGBM 있는 같은 시점만:"); print(m10.groupby("method").agg(MAE=("err", "mean"), 일치=("hit3", "mean"), n=("err", "size")).round(3).to_string())
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
        names = list(runs)
        fig, ax = plt.subplots(1, len(names), figsize=(4 * len(names), 3.4), squeeze=False)
        for i, nm in enumerate(names):
            rd, _ = real_metrics(runs[nm]); T = int(runs[nm]["traj"].fr.max() // FPS) + 1
            ax[0, i].plot(range(T), rd[:T], "k-", lw=2, label="실제")
            for meth, sty in (("시뮬(GA v2, 유입 oracle)", "C3-"), ("시뮬(GA v2, 유입 rate)", "C1-"), ("직전값", "C7:")):
                q = A[(A.run == nm) & (A.h == 20) & (A.method == meth)].sort_values("t")
                ax[0, i].plot(q.t + 20, q.pred, sty, marker="o", ms=3, label=f"{meth} 20초 전 예측")
            ax[0, i].axhline(CAUTION, color="orange", lw=0.8); ax[0, i].set_title(f"{nm} (폭 {runs[nm]['w']}m)", fontsize=9)
            ax[0, i].set_xlabel("초"); ax[0, i].set_ylabel("문 앞 밀도(명/㎡)")
        ax[0, 0].legend(fontsize=6); fig.tight_layout(); fig.savefig(DATA / "simfc.png", dpi=100)
    except Exception as e:
        print("그림 생략:", e)


if __name__ == "__main__":
    main()
