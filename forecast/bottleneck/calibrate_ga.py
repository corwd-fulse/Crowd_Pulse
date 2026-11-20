"""JuPedSim 시뮬레이션 파라미터를 유전 알고리즘(GA)으로 실제 병목 실험 궤적에 맞춘다 (명세서 FR-SIM-02 준비).
데이터: 위리히 'Crowds in front of bottlenecks'(Wuppertal 2018, CC BY 4.0, DOI 10.34735/ped.2018.1), 문 폭 0.5m, 회차별 통로 형상(WKT).
- 시뮬레이션: CollisionFreeSpeedModelV2. 사람마다 실제로 처음 보인 시각·위치에 넣고, 문 아래 출구로 보낸다.
- 비교 지표(초 단위, crowdqueue.zone_seconds와 같은 정의): 문 앞(0≤y<2m, 통로 안) 밀도, 문 통과(처음 y<0) 누적 인원.
- 적합도(작을수록 좋음) = 밀도 MAE(명/㎡) + 누적 통과 인원 MAE / 참가자 수. 회차 평균.
- GA: 실수 유전자 5개, 토너먼트 선택, BLX-α 교차, 가우시안 변이, 엘리트 보존 (Johansson·Helbing·Shukla 2007과 같은 '실측 궤적에 진화 알고리즘으로 맞추기').
- 학습/검증: 동기 h0 회차만, 통로 폭마다 회차를 번갈아 학습·검증으로 나눔. 기본값(JuPedSim 기본 파라미터)과 GA 결과를 검증 회차에서 비교.
실행(data-area 루트): C:/Users/Woori/venvs/nf/Scripts/python.exe -m forecast.bottleneck.calibrate_ga [--pop 20 --gen 12]
출력: data/crowdqueue/ga_history.csv, ga_compare.csv, ga_best.json, ga_compare.png"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
import jupedsim as jps
import shapely
from shapely.ops import nearest_points

from .crowdqueue import DATA, FPS, FRONT_DEPTH, CAUTION, HOLD_S, bottleneck_label

GENES = {  # 이름: (하한, 상한, JuPedSim 기본값)
    "desired_speed": (0.6, 2.0, 1.2),
    "time_gap": (0.3, 2.0, 1.0),
    "radius": (0.12, 0.30, 0.2),
    "strength_neighbor_repulsion": (2.0, 15.0, 8.0),
    "range_neighbor_repulsion": (0.05, 0.5, 0.1),
    "speed_cv": (0.0, 0.3, 0.0),  # 사람마다 희망 속도 차이(변동계수). 0 = 모두 같은 속도(v1·v2와 같음)
}
AGENT_GENES = [k for k in GENES if k != "speed_cv"]
LO, HI = np.array([g[0] for g in GENES.values()]), np.array([g[1] for g in GENES.values()])
DEFAULT = np.array([g[2] for g in GENES.values()])
EXIT = [(-0.6, -3.0), (0.6, -3.0), (0.6, -2.4), (-0.6, -2.4)]  # 문(y -0.15 ~ -1.1) 아래 출구
EXTRA_S = 30  # 실제 기록이 끝난 뒤 시뮬레이션을 더 돌리는 최대 시간


def load_h0_runs():
    meta = json.load(open(DATA / "2018crowdqueue.json", encoding="utf-8"))["experiment"]["run"]
    runs = []
    for m in meta:
        if m["parameter"]["motivation"] != "h0":
            continue
        f = next((DATA / "txt").glob(m["run_name"][:3] + "*.txt"), None)
        if f is None:
            continue
        d = pd.read_csv(f, sep=r"\s+", comment="#", header=None, names=["id", "fr", "x", "y", "z"])
        runs.append({"name": m["run_name"], "w": float(m["parameter"]["corridor_width"]), "wkt": m["wkt_geometry"], "traj": d})
    return sorted(runs, key=lambda r: r["name"])


def seconds_metrics(pos, w, n_secs):
    """pos: DataFrame[sec, id, x, y] (각 초의 위치) -> (밀도[n_secs], 누적 통과[n_secs]). 통과 = 처음 y<0 인 초."""
    area = w * FRONT_DEPTH
    front = pos[(pos.y >= 0) & (pos.y < FRONT_DEPTH) & (pos.x.abs() <= w / 2)]
    dens = front.groupby("sec").id.nunique().reindex(range(n_secs), fill_value=0).to_numpy() / area
    passed = pos[pos.y < 0].groupby("id").sec.min().value_counts().reindex(range(n_secs), fill_value=0).cumsum().to_numpy()
    return dens, passed


def real_metrics(run):
    d = run["traj"]
    n_secs = int(d.fr.max() // FPS) + 1 + EXTRA_S
    p = d[d.fr % FPS == 0].assign(sec=lambda x: x.fr // FPS)
    return seconds_metrics(p, run["w"], n_secs)


def simulate(run, g):
    """파라미터 g(GENES 순서) -> 각 초의 위치 DataFrame[sec, id, x, y]."""
    prm = dict(zip(GENES, g))
    cv = prm.pop("speed_cv", 0.0)
    rng = np.random.default_rng(sum(run["name"].encode()))  # 회차마다 고정된 난수(같은 파라미터면 같은 결과)
    area = shapely.from_wkt(run["wkt"])
    sim = jps.Simulation(model=jps.CollisionFreeSpeedModelV2(), geometry=area, dt=0.01)
    ex = sim.add_exit_stage(EXIT)
    jr = sim.add_journey(jps.JourneyDescription([ex]))
    d = run["traj"]
    first = d.sort_values("fr").groupby("id").first()
    spawn = sorted(((r.fr / FPS, pid, r.x, r.y) for pid, r in first.iterrows()))
    z = np.clip(rng.standard_normal(len(spawn)), -2, 2)
    v_of = {s[1]: prm["desired_speed"] * max(0.3, 1 + cv * zi) for s, zi in zip(spawn, z)}
    inner = area.buffer(-prm["radius"] - 0.02)
    n_secs = int(d.fr.max() // FPS) + 1 + EXTRA_S
    rows, k, id_map, pending = [], 0, {}, []
    offs = [(0.0, 0.0)] + [(r * np.cos(a), r * np.sin(a)) for r in (0.1, 0.2, 0.3) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]

    def try_add(pid, x, y):
        """실제 위치에 넣고, 다른 사람과 겹치면 0.1~0.3m 주변 빈자리. 그래도 안 되면 False(잠시 뒤 다시 시도)."""
        for dx, dy in offs:
            pt = shapely.Point(x + dx, y + dy)
            if not inner.contains(pt):
                pt = nearest_points(inner, pt)[0]
            try:
                aid = sim.add_agent(jps.CollisionFreeSpeedModelV2AgentParameters(position=(pt.x, pt.y), journey_id=jr, stage_id=ex,
                                                                                         **{**prm, "desired_speed": v_of[pid]}))
                id_map[aid] = pid
                return True
            except Exception:
                continue
        return False

    for s in range(n_secs):
        while k < len(spawn) and spawn[k][0] <= s:  # 이 초까지 나타난 사람(벽 안쪽으로 살짝 보정)
            pending.append(spawn[k][1:]); k += 1
        for a in sim.agents():
            rows.append((s, id_map.get(a.id, a.id), a.position[0], a.position[1]))
        if k >= len(spawn) and not pending and sim.agent_count() == 0:
            break
        for step in range(100):  # dt 0.01 x 100 = 1초. 못 넣은 사람은 0.1초마다 다시 시도
            if pending and step % 10 == 0:
                pending = [p for p in pending if not try_add(*p)]
            sim.iterate()
    return pd.DataFrame(rows, columns=["sec", "id", "x", "y"]), n_secs


FIT = {"v": 1}  # 적합도 버전. 2 = v1 + 최고 밀도 오차 + 3명/㎡ 이상 지속 시간 오차(과대 추정 억제)
W_PEAK, W_DUR = 0.3, 1.0


def peak3(d):
    """3초 이동 평균의 최고값(1초 튐을 빼고 본 최고 밀도)."""
    return float(np.convolve(d, np.ones(3) / 3, "valid").max()) if len(d) >= 3 else float(np.max(d))


def score_run(run, g, real):
    pos, n_secs = simulate(run, g)
    dens, passed = seconds_metrics(pos, run["w"], n_secs)
    rd, rp = real
    n = run["traj"].id.nunique()
    s = np.abs(dens - rd).mean() + np.abs(passed - rp).mean() / n
    if FIT["v"] == 2:
        t_real = int(run["traj"].fr.max() // FPS) + 1
        s += W_PEAK * abs(peak3(dens) - peak3(rd)) + W_DUR * abs(int((dens >= CAUTION).sum()) - int((rd >= CAUTION).sum())) / t_real
    return float(s), dens, passed


def fitness(runs, reals, g):
    return float(np.mean([score_run(r, g, rl)[0] for r, rl in zip(runs, reals)]))


_W = {}  # 병렬 작업 프로세스마다 한 번 읽어 둔 회차


def _init(names, fit_v=1):
    FIT["v"] = fit_v
    runs = [r for r in load_h0_runs() if r["name"] in names]
    _W["runs"], _W["reals"] = runs, [real_metrics(r) for r in runs]


def _fit(g):
    return fitness(_W["runs"], _W["reals"], np.asarray(g))


def ga(runs, reals, pop=20, gen=12, seed=0, elite=2, alpha=0.3, pm=0.2, log=print, workers=None, seeds=()):
    from concurrent.futures import ProcessPoolExecutor
    import os
    rng = np.random.default_rng(seed)
    ex = ProcessPoolExecutor(workers or max(1, (os.cpu_count() or 2) - 1), initializer=_init, initargs=([r["name"] for r in runs], FIT["v"]))
    evaluate = lambda G: np.array(list(ex.map(_fit, [list(g) for g in G])))
    S0 = np.vstack([DEFAULT, *seeds])  # 기본값(과 이전 결과)도 처음 집단에 넣음
    P = np.vstack([S0, LO + rng.random((pop - len(S0), len(LO))) * (HI - LO)])
    F = evaluate(P)
    hist = []
    for gi in range(gen + 1):
        order = np.argsort(F); P, F = P[order], F[order]
        hist.append({"generation": gi, "best": F[0], "median": float(np.median(F)), **dict(zip(GENES, P[0]))})
        log(f"세대 {gi}: 최고 {F[0]:.4f}, 중앙 {np.median(F):.4f}, " + ", ".join(f"{k}={v:.3f}" for k, v in zip(GENES, P[0])))
        if gi == gen:
            break
        pick = lambda: P[min(rng.integers(0, pop, 3))]  # 토너먼트(3): 정렬돼 있으니 가장 작은 번호가 승자
        kids = []
        while len(kids) < pop - elite:
            a, b = pick(), pick()
            lo, hi = np.minimum(a, b), np.maximum(a, b); span = hi - lo
            c = rng.uniform(lo - alpha * span, hi + alpha * span)                        # BLX-α 교차
            c = np.where(rng.random(len(c)) < pm, c + rng.normal(0, 0.1 * (HI - LO)), c)  # 가우시안 변이
            kids.append(np.clip(c, LO, HI))
        K = np.array(kids)
        P = np.vstack([P[:elite], K]); F = np.concatenate([F[:elite], evaluate(K)])
    ex.shutdown()
    return P[0], pd.DataFrame(hist)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pop", type=int, default=20); ap.add_argument("--gen", type=int, default=12)
    ap.add_argument("--fitness", type=int, default=1, choices=(1, 2), help="2 = 최고 밀도·3명/㎡ 이상 지속 시간 오차 추가, 결과는 *_v2 파일")
    ap.add_argument("--tag", help="결과 파일 접미어(예: v3 -> ga_*_v3)")
    ap.add_argument("--seed-from", help="처음 집단에 넣을 이전 결과 json(예: ga_best_v2.json)")
    a = ap.parse_args()
    FIT["v"] = a.fitness
    sfx = f"_{a.tag}" if a.tag else ("" if a.fitness == 1 else f"_v{a.fitness}")
    seed_file = DATA / (a.seed_from or "ga_best.json")
    prev = json.loads(seed_file.read_text(encoding="utf-8"))["best"] if (a.fitness != 1 or a.seed_from) and seed_file.exists() else None
    seeds = [np.array([prev.get(k, GENES[k][2]) for k in GENES])] if prev else []  # 이전 결과에 없는 유전자는 기본값
    seed_lab = "GA " + (seed_file.stem.split("_")[-1] if seed_file.stem != "ga_best" else "v1")
    runs = load_h0_runs()
    by_w = {}
    for r in runs:
        by_w.setdefault(r["w"], []).append(r)
    train = [r for rs in by_w.values() for i, r in enumerate(rs) if i % 2 == 0]
    test = [r for rs in by_w.values() for i, r in enumerate(rs) if i % 2 == 1]
    print("학습 회차:", [r["name"] for r in train]); print("검증 회차:", [r["name"] for r in test], flush=True)
    reals = {r["name"]: real_metrics(r) for r in runs}
    t0 = time.time(); f0 = fitness(train, [reals[r["name"]] for r in train], DEFAULT)
    print(f"기본값 학습 적합도 {f0:.4f} (평가 1회 {time.time() - t0:.1f}초)", flush=True)
    best, hist = ga(train, [reals[r["name"]] for r in train], a.pop, a.gen, log=lambda s: print(s, flush=True), seeds=seeds)
    hist.to_csv(DATA / f"ga_history{sfx}.csv", index=False)
    (DATA / f"ga_best{sfx}.json").write_text(json.dumps({"best": dict(zip(GENES, map(float, best))), "default": dict(zip(GENES, map(float, DEFAULT))),
                                                   "train": [r["name"] for r in train], "test": [r["name"] for r in test]}, ensure_ascii=False, indent=1), encoding="utf-8")
    rows, curves = [], {}
    for split, rs in (("학습", train), ("검증", test)):
        for r in rs:
            rd, rp = reals[r["name"]]
            n_real_end = int(r["traj"].fr.max() // FPS) + 1
            for label, g in (("기본값", DEFAULT), *(((seed_lab, seeds[0]),) if seeds else ()), ("GA" + sfx.replace("_", " "), best)):
                sc, dens, passed = score_run(r, g, reals[r["name"]])
                done = np.argmax(passed >= passed.max()) if passed.max() > 0 else np.nan
                rows.append({"split": split, "run": r["name"], "w": r["w"], "n": r["traj"].id.nunique(), "params": label, "fitness": sc,
                             "dens_MAE": np.abs(dens - rd).mean(), "max_dens_real": rd.max(), "max_dens_sim": dens.max(),
                             "s_over3_real": int((rd >= CAUTION).sum()), "s_over3_sim": int((dens >= CAUTION).sum()),
                             "bottleneck_real": bool(bottleneck_label(rd).any()), "bottleneck_sim": bool(bottleneck_label(dens).any()),
                             "all_passed_real_s": int(np.argmax(rp >= rp.max())), "all_passed_sim_s": int(done) if done == done else None,
                             "passed_sim/real": passed.max() / max(rp.max(), 1)})
                curves[(r["name"], label)] = (dens, passed)
            curves[(r["name"], "실제")] = (rd, rp)
    C = pd.DataFrame(rows).round(3); C.to_csv(DATA / f"ga_compare{sfx}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(C.to_string(index=False))
    C["peak_err"] = (C.max_dens_sim - C.max_dens_real).abs(); C["over3_err"] = (C.s_over3_sim - C.s_over3_real).abs()
    C["bn_match"] = C.bottleneck_sim == C.bottleneck_real
    print(C.groupby(["split", "params"], sort=False)[["fitness", "dens_MAE", "peak_err", "over3_err", "bn_match"]].mean().round(4).to_string())
    labels = [lab for lab in ("기본값", seed_lab if seeds else "", "GA" + sfx.replace("_", " ")) if (test[0]["name"], lab) in curves]
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(2, len(test), figsize=(4 * len(test), 6), squeeze=False)
        for i, r in enumerate(test):
            for lab, sty in zip(["실제"] + labels, ["k-", "C0--", "C1-", "C3-"] if len(labels) == 3 else ["k-", "C0--", "C3-"]):
                dd, pp = curves[(r["name"], lab)]
                ax[0, i].plot(dd, sty, label=lab); ax[1, i].plot(pp, sty, label=lab)
            ax[0, i].axhline(CAUTION, color="orange", lw=0.8); ax[0, i].set_title(f"{r['name']} (폭 {r['w']}m)")
            ax[0, i].set_ylabel("문 앞 밀도(명/㎡)"); ax[1, i].set_ylabel("통과 누적(명)"); ax[1, i].set_xlabel("초")
        ax[0, 0].legend(); fig.tight_layout(); fig.savefig(DATA / f"ga_compare{sfx}.png", dpi=110)
    except Exception as e:
        print("그림 생략:", e)


if __name__ == "__main__":
    main()
