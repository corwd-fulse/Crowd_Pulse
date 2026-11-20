"""JuPedSim으로 구역 혼잡 학습 데이터 생성: 통로 폭·출구 폭·형태·유입량이 다른 1시간 시나리오, 1초 기록.

- 형태: straight(직선 통로, 출구 = 통로 전체 폭), bottleneck(방 끝에 좁은 문 + 2m 복도).
- 구역 = 방(x < L). 기록: 인원 n, 유입 inflow, 유출 outflow, 평균 속도 speed, 출구 방향 성분 heading(cos), 밀도 density.
- 유입량 λ(t)(명/초) = 바닥 + 가우시안 덩어리(행사 끝·몰림). 출구 용량 1.3명/m/s 기준으로 0.1~1.6배.
- 보행 파라미터 = GA v2 보정값(위리히 병목 실험, data/crowdqueue/ga_best_v2.json).
한계: 모델은 실제 사람이 아니라 여기서 정한 유입 패턴·보행 모델을 배운다 → 모델 비교·공간 특성 효과 확인용.
실행(data-area 루트): C:/Users/Woori/venvs/nf/Scripts/python.exe -m forecast.bottleneck.sim_dataset --n 120 --workers 7
출력: data/simcrowd/seconds.parquet(시나리오×초), scenarios.csv"""
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd
import jupedsim as jps
import shapely

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "simcrowd"
PRM = json.loads((ROOT / "data" / "crowdqueue" / "ga_best_v2.json").read_text())["best"]
CAP = 1.3  # 출구 용량(명/m/s), Weidmann 근사 — 원문 미확인
L, DUR = 30.0, 3600


N_TWO_SHAPES = 120  # 0~119 = 직선/병목 번갈아(처음 만든 세트), 120~179 = T자 합류, 180~ = 한산(직선/병목) — 번호별 시나리오가 바뀌지 않게 고정
N_TJ_END = 180


def make_scenario(i, seed=0):
    rng = np.random.default_rng([seed, i])
    if i >= N_TJ_END:
        return make_quiet(i, rng)
    if i >= N_TWO_SHAPES:
        return make_tjunction(i, rng)
    shape = ["straight", "bottleneck"][i % 2]
    W = float(rng.uniform(1.5, 10))
    d = W if shape == "straight" else float(rng.uniform(0.8, min(3.0, W - 0.5)))
    cap = CAP * d
    k = int(rng.integers(1, 5))
    bumps = [(float(rng.uniform(0, DUR)), float(rng.uniform(120, 600)), float(rng.uniform(0.3, 1.5) * cap)) for _ in range(k)]
    return dict(sid=i, shape=shape, width=round(W, 3), exit_width=round(d, 3), area=round(L * W, 2), base=round(float(rng.uniform(0.1, 0.4) * cap), 4),
                bumps=json.dumps([[round(a, 1), round(b, 1), round(c, 4)] for a, b, c in bumps]))


def make_quiet(i, rng):
    """한산한 구역(실제 광장·골목 영상 수준): 평소 유입 = 출구 용량의 0.1~10%, 몰림 0~3번(용량의 2~60%).
    11-20 광화문 영상 유입이 기존 시나리오 하위 1%보다 약 100배 작아 모델이 학습 범위 밖이었던 문제를 메우기 위함."""
    shape = ["straight", "bottleneck"][i % 2]
    W = float(rng.uniform(1.5, 10))
    d = W if shape == "straight" else float(rng.uniform(0.8, min(3.0, W - 0.5)))
    cap = CAP * d
    k = int(rng.integers(0, 4))
    bumps = [(float(rng.uniform(0, DUR)), float(rng.uniform(120, 600)), float(10 ** rng.uniform(np.log10(0.02), np.log10(0.6)) * cap)) for _ in range(k)]
    return dict(sid=i, shape=shape, width=round(W, 3), exit_width=round(d, 3), area=round(L * W, 2),
                base=round(float(10 ** rng.uniform(-3, -1) * cap), 5), bumps=json.dumps([[round(a, 1), round(b, 1), round(c, 4)] for a, b, c in bumps]))


ARM = 15.0  # T자: 가로 통로 한쪽 길이(m)


def make_tjunction(i, rng):
    """가로 통로(폭 Wb) 양 끝에서 들어와 가운데서 합쳐져 세로 통로(폭 W, 길이 L) 끝 출구로. 좌·우 유입 따로."""
    W, Wb = float(rng.uniform(1.5, 8)), float(rng.uniform(1.5, 6))
    cap = CAP * W
    side = {}
    for s in ("", "_r"):
        k = int(rng.integers(1, 4))
        side["base" + s] = round(float(rng.uniform(0.05, 0.2) * cap), 4)
        side["bumps" + s] = json.dumps([[round(float(rng.uniform(0, DUR)), 1), round(float(rng.uniform(120, 600)), 1), round(float(rng.uniform(0.2, 1.0) * cap), 4)]
                                        for _ in range(k)])
    return dict(sid=i, shape="tjunction", width=round(W, 3), exit_width=round(W, 3), branch_width=round(Wb, 3), area=round(W * (L + Wb), 2), **side)


def rate(sc, t, s=""):
    lam = sc["base" + s] + sum(h * np.exp(-0.5 * ((t - c) / w) ** 2) for c, w, h in json.loads(sc["bumps" + s]))
    return min(lam, 1.8 * sc.get("branch_width", sc["width"]))  # 입구 폭으로 들어올 수 있는 한계


def layout(sc):
    """형태 -> (걸을 수 있는 공간, 출구 다각형 꼭짓점, 구역 사각형). 시뮬레이션과 뷰어가 같이 쓴다."""
    W = sc["width"]
    if sc["shape"] == "tjunction":
        x0, x1, top = ARM, ARM + W, L + sc["branch_width"]
        geo = shapely.union(shapely.box(x0, 0, x1, L + 0.01), shapely.box(0, L, 2 * ARM + W, top))
        return geo, [(x0, 0), (x1, 0), (x1, 0.5), (x0, 0.5)], shapely.box(x0, 0, x1, top)
    room = shapely.box(0, 0, L, W)
    if sc["shape"] == "straight":
        return room, [(L - 0.5, 0), (L, 0), (L, W), (L - 0.5, W)], room
    d = sc["exit_width"]; y0 = (W - d) / 2
    geo = shapely.union(room, shapely.box(L - 0.01, y0, L + 2, y0 + d))
    return geo, [(L + 1.5, y0), (L + 2, y0), (L + 2, y0 + d), (L + 1.5, y0 + d)], room


def simulate_tjunction(sc, dt=0.01, dur=DUR, traj=None):
    W, Wb = sc["width"], sc["branch_width"]
    x0, x1, top = ARM, ARM + W, L + Wb
    geo, exit_poly, _ = layout(sc)
    sim = jps.Simulation(model=jps.CollisionFreeSpeedModelV2(), geometry=geo, dt=dt)
    ex = sim.add_exit_stage(exit_poly)
    jr = sim.add_journey(jps.JourneyDescription([ex]))
    rng = np.random.default_rng(sc["sid"] + 10_000)
    per_s, r = int(round(1 / dt)), PRM["radius"]
    target = np.array([(x0 + x1) / 2, 0.0])
    in_zone = lambda p: x0 <= p[0] <= x1 and p[1] < top  # 합류 지점 + 세로 통로
    origin, prev, pend, rows = {}, {}, {"": 0, "_r": 0}, []
    for t in range(dur):
        for s, xs in (("", lambda: r + 0.05 + rng.uniform(0, 0.6)), ("_r", lambda: 2 * ARM + W - r - 0.05 - rng.uniform(0, 0.6))):
            pend[s] += rng.poisson(rate(sc, t, s))
            fails, k = 0, 0
            while k < min(pend[s], int(2 * Wb) + 2) and fails < 3:
                try:
                    aid = sim.add_agent(jps.CollisionFreeSpeedModelV2AgentParameters(
                        position=(xs(), rng.uniform(L + r + 0.05, top - r - 0.05)), journey_id=jr, stage_id=ex, **PRM))
                    origin[aid] = s; pend[s] -= 1
                except Exception:
                    fails += 1
                k += 1
        for _ in range(per_s):
            sim.iterate()
        if traj is not None:  # 뷰어용: 모든 사람 위치(1초마다)
            traj.extend((t, a.id, a.position[0], a.position[1]) for a in sim.agents())
        pos = {a.id: a.position for a in sim.agents() if in_zone(a.position)}
        new, gone = set(pos) - set(prev), set(prev) - set(pos)
        sp, hd = [], []
        for aid, p in pos.items():
            if aid in prev:
                v = np.subtract(p, prev[aid]); s_ = float(np.hypot(*v)); sp.append(s_)
                if s_ > 1e-3:
                    g = target - p; hd.append(float(v @ g / (s_ * np.hypot(*g))))
        n = len(pos)
        rows.append((sc["sid"], t, n, len(new), len(gone), np.mean(sp) if sp else np.nan, np.mean(hd) if hd else np.nan, n / sc["area"],
                     pend[""] + pend["_r"], sum(origin[a] == "" for a in new), sum(origin[a] == "_r" for a in new)))
        prev = pos
    df = pd.DataFrame(rows, columns=["sid", "t", "n", "inflow", "outflow", "speed", "heading", "density", "queue_outside", "in_left", "in_right"])
    assert (df.inflow == df.in_left + df.in_right).all() and (df.n.diff().iloc[1:] == (df.inflow - df.outflow).iloc[1:]).all(), "인원 보존 위반"
    return df


def simulate(sc, dt=0.01, dur=DUR, traj=None):  # GA 보정과 같은 dt. 0.05는 인원이 20~30% 많게 나옴(11-19 확인)
    if sc["shape"] == "tjunction":
        return simulate_tjunction(sc, dt, dur, traj)
    W = sc["width"]
    geo, exit_poly, _ = layout(sc)
    sim = jps.Simulation(model=jps.CollisionFreeSpeedModelV2(), geometry=geo, dt=dt)
    ex = sim.add_exit_stage(exit_poly)
    jr = sim.add_journey(jps.JourneyDescription([ex]))
    rng = np.random.default_rng(sc["sid"] + 10_000)
    per_s, r = int(round(1 / dt)), PRM["radius"]
    target = np.array([L + 2, W / 2])
    prev, pending, rows = {}, 0, []
    for t in range(dur):
        pending += rng.poisson(rate(sc, t))  # 못 들어간 사람은 입구 밖 대기(구역 인원 아님)
        added, fails = 0, 0
        for _ in range(min(pending, int(2 * W) + 2)):  # 입구 폭만큼만 시도: 대기가 수백 명 쌓여도 초당 시도 횟수 고정
            try:
                sim.add_agent(jps.CollisionFreeSpeedModelV2AgentParameters(
                    position=(r + 0.05 + rng.uniform(0, 0.6), rng.uniform(r + 0.05, W - r - 0.05)), journey_id=jr, stage_id=ex, **PRM))
                added += 1
            except Exception:
                fails += 1
                if fails >= 3:  # 입구가 꽉 참
                    break
        pending -= added
        before = {a.id for a in sim.agents() if a.position[0] < L} | set(prev)
        for _ in range(per_s):
            sim.iterate()
        if traj is not None:  # 뷰어용: 모든 사람 위치(1초마다)
            traj.extend((t, a.id, a.position[0], a.position[1]) for a in sim.agents())
        pos = {a.id: a.position for a in sim.agents() if a.position[0] < L}
        out = len(before - set(pos))
        sp, hd = [], []
        for aid, p in pos.items():
            if aid in prev:
                v = np.subtract(p, prev[aid]); s = float(np.hypot(*v)); sp.append(s)
                if s > 1e-3:
                    g = target - p; hd.append(float(v @ g / (s * np.hypot(*g))))
        n = len(pos)
        rows.append((sc["sid"], t, n, added, out, np.mean(sp) if sp else np.nan, np.mean(hd) if hd else np.nan, n / sc["area"], pending))
        prev = pos
    df = pd.DataFrame(rows, columns=["sid", "t", "n", "inflow", "outflow", "speed", "heading", "density", "queue_outside"])
    df["in_left"], df["in_right"] = df.inflow, 0  # 한 방향 유입 = 전부 '왼쪽'(T자와 열 맞춤)
    # 보존 검사: 지금 인원 = 직전 인원 + 유입 - 유출
    assert (df.n.diff().iloc[1:] == (df.inflow - df.outflow).iloc[1:]).all(), "인원 보존 위반"
    return df


def _run(sc):
    f = OUT / "parts" / f"{sc['sid']:03d}.parquet"
    if f.exists():  # 중간에 멈췄다 다시 돌리면 끝난 시나리오는 건너뜀
        return pd.read_parquet(f), 0.0
    t0 = time.time(); d = simulate(sc)
    d.to_parquet(f, index=False)
    return d, time.time() - t0


def _record(sc):
    tr = []
    d = simulate(sc, traj=tr)
    ref = OUT / "parts" / f"{sc['sid']:03d}.parquet"
    if ref.exists():  # 같은 난수면 같은 결과여야 함
        assert (pd.read_parquet(ref).n.to_numpy() == d.n.to_numpy()).all(), f"sid {sc['sid']} 재현 안 됨"
    pd.DataFrame(tr, columns=["t", "id", "x", "y"]).astype({"x": "float32", "y": "float32"}).to_parquet(OUT / "traj" / f"{sc['sid']:03d}.parquet", index=False)
    geo, exit_poly, zone = layout(sc)
    (OUT / "traj" / f"{sc['sid']:03d}.json").write_text(json.dumps({"geometry": geo.wkt, "exit": shapely.Polygon(exit_poly).wkt, "zone": zone.wkt}))
    print(f"sid {sc['sid']} 위치 {len(tr):,}행 저장", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--start", type=int, default=0, help="이 번호부터만 시뮬레이션(다른 실행과 겹치지 않게). scenarios.csv는 0~n-1 전부 기록")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--record", help="뷰어용 위치 기록: 시나리오 번호 쉼표 목록(이미 만든 시나리오를 같은 난수로 다시 돌림)")
    a = ap.parse_args()
    if a.record:
        (OUT / "traj").mkdir(parents=True, exist_ok=True)
        scs = pd.read_csv(OUT / "scenarios.csv").set_index("sid")
        with ProcessPoolExecutor(a.workers) as pool:
            list(pool.map(_record, [dict(scs.loc[int(s)].dropna(), sid=int(s)) for s in a.record.split(",")]))
        return
    (OUT / "parts").mkdir(parents=True, exist_ok=True)
    scs = [make_scenario(i) for i in range(a.n)]
    pd.DataFrame(scs).to_csv(OUT / "scenarios.csv", index=False)
    scs = scs[a.start:]
    parts = []
    with ProcessPoolExecutor(a.workers) as pool:
        for k, (d, s) in enumerate(pool.map(_run, scs)):
            parts.append(d)
            print(f"[{k + 1}/{len(scs)}] sid {d.sid.iloc[0]} {scs[k]['shape']} W {scs[k]['width']:.1f} d {scs[k]['exit_width']:.1f} 최대 {d.n.max()}명 "
                  f"최고 밀도 {d.density.max():.2f} {s:.0f}s", flush=True)
    if a.start == 0:  # 부분 실행이면 전체 묶음 파일은 건드리지 않음
        pd.concat(parts).to_parquet(OUT / "seconds.parquet", index=False)


if __name__ == "__main__":
    main()
