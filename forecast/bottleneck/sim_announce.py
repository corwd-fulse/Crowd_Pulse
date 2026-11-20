"""공지 있음/없음 테스트: 같은 테스트 시나리오를 '몰림이 공지됨'과 '공지 안 됨' 두 조건으로 예측해 정확도와 경보를 비교.

모델(LightGBM, sim_models와 같은 설정·분할):
- 기준: 공지 특징 없이 학습
- 전부 공지 학습: 학습 몰림이 모두 공지됨
- 일부 숨김 학습: 학습 몰림의 --hide 비율(기본 40%)을 공지 안 됨으로 숨겨 학습 (공지 없는 몰림도 배우게)
- 기준 + 공지 보정: 기준 예측 + a_h × (공지대로면 출구 용량을 넘어 쌓일 양) + b_h × (유입률 변화 × 통과 시간).
  a_h, b_h는 검증 잔차에 최소제곱(음수는 0). (단순히 예상 유입을 더하는 방식은 검증 w가 모두 0이라 버림 — 대부분 그냥 지나감.)
  공지가 없으면 보정 0 = 기준 모델과 같음.
테스트 조건: 공지 있음(모든 몰림 공지, 오차 ±3분·±30%) / 일부 공지(--hide 비율 숨김) / 공지 없음(평소 유입만 알려 줌).
- 전환: 그 시나리오에 공지된 몰림이 있으면 공지 학습 모델, 하나도 없으면 기준 모델.

경보: 예측 시점(30초마다) 5·10·15분 뒤 예측 밀도 중 하나라도 기준 이상이면 경보.
사건 = 실제 밀도가 기준 이상으로 20초(10초 칸 2개) 이상 이어진 구간의 시작. 사건 전 15분 안에 경보가 있으면 잡음, 선행 시간 = 사건 시작 - 첫 경보.
헛경보 = 지금 기준 미만이고 앞으로 15분 안에 사건이 없는데 울린 경보(그런 시점 중 비율).
비교 기준 '지금 밀도 경보' = 지금 밀도가 기준 이상일 때 울림(예측 없이 보고 반응).
실행(data-area 루트): C:/Users/Woori/venvs/nf/Scripts/python.exe -m forecast.bottleneck.sim_announce --out 20261120_120_announce
"""
import argparse, json, time

import numpy as np, pandas as pd

from .sim_dataset import L, PRM
from .sim_models import H, RUNS, STEP, build, fit_lgbm, load, md_table, noisy_plan

THRS = (3.0, 2.0)  # 명/㎡: 행안부 '주의'(3) / 참고(2)
WIN = 900  # 경보가 사건보다 이만큼(초) 안쪽으로 먼저면 잡은 것


def events(dens, t, thr):
    """10초 칸 밀도 -> 사건 시작 시각들(기준 이상 2칸 이상 연속인 구간의 첫 칸 끝 시각)."""
    hi = dens >= thr
    out, i = [], 0
    while i < len(hi):
        if hi[i]:
            j = i
            while j < len(hi) and hi[j]:
                j += 1
            if j - i >= 2:
                out.append(t[i])
            i = j
        else:
            i += 1
    return np.array(out)


def alarm_stats(alarm, ev_by_sid, cur_hi, M):
    """alarm: 예측 시점마다 True/False. -> 사건 수, 잡은 수, 선행 시간 목록, 헛경보 비율."""
    caught, leads, n_ev, fa, fa_base = 0, [], 0, 0, 0
    for sid, d in M.assign(a=alarm, hi=cur_hi).groupby("sid"):
        ev = ev_by_sid.get(sid, np.array([]))
        ev = ev[(ev > d.t.min()) & (ev <= d.t.max() + STEP)]  # 예측 시점이 덮는 사건만
        for e in ev:
            n_ev += 1
            w = d[(d.t >= e - WIN) & (d.t < e) & d.a]
            if len(w):
                caught += 1; leads.append((e - w.t.min()) / 60)
        quiet = ~d.hi.to_numpy() & np.array([not ((ev > x) & (ev <= x + WIN)).any() for x in d.t])
        fa += (d.a.to_numpy() & quiet).sum(); fa_base += quiet.sum()
    return n_ev, caught, leads, fa / max(fa_base, 1)


def phys_terms(sci, M, cap, hide=0.0):
    """예측 시점마다 [쌓임, 통과] 두 값 (시간 구간별 리스트, 각 (N, 2)). 공지 몰림만의 효과 = 공지 계획 - 평소 유입만."""
    out = [np.zeros((len(M), 2)) for _ in H]
    for sid, idx in M.groupby("sid").indices.items():
        t, c = M.t.to_numpy()[idx].astype(float), cap[idx][0]
        for k, h in enumerate(H):
            vals = []
            for hd in (hide, 1.0):
                base, bumps = noisy_plan(sci.loc[sid], hide=hd)
                lam = lambda x: base + np.zeros_like(x) + sum(hh * np.exp(-0.5 * ((x - cc) / w) ** 2) for cc, w, hh in bumps)
                grid = t[:, None] + np.arange(1, h + 1)
                vals.append(np.stack([np.clip(lam(grid) - c, 0, None).sum(1), (lam(t + h) - lam(t)) * L / PRM["desired_speed"]], 1))
            out[k][idx] = vals[0] - vals[1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hide", type=float, default=0.4, help="일부 숨김 학습에서 숨길 몰림 비율")
    ap.add_argument("--test-hide", type=float, default=0.4, help="테스트 '일부 공지' 조건에서 숨길 비율(학습 비율을 바꿔 비교할 때 고정)")
    ap.add_argument("--shapes", default="straight,bottleneck", help="쓸 형태(쉼표). T자까지: straight,bottleneck,tjunction")
    a = ap.parse_args()
    t0 = time.time()
    run = RUNS / a.out
    if run.exists() and any(run.iterdir()):
        raise SystemExit(f"{run} 가 이미 있어요. 다른 --out 이름을 쓰세요.")
    run.mkdir(parents=True, exist_ok=True)

    g, sc = load()
    sc = sc[sc["shape"].isin(a.shapes.split(","))]; g = g[g.sid.isin(sc.sid)]
    sci = sc.set_index("sid", drop=False)
    F0, X, Y, N0, M = build(g)
    Fa = build(g, sci)[0]                   # 전부 공지
    Fh = build(g, sci, hide=1.0)[0]         # 공지 없음
    Fp = build(g, sci, hide=a.hide)[0]      # 일부 숨김
    rng = np.random.default_rng(0)  # sim_models와 같은 분할
    sids = rng.permutation(sc.sid.to_numpy())
    ntr, nva = int(len(sids) * 0.7), int(len(sids) * 0.15)
    part = {s: "train" for s in sids[:ntr]} | {s: "val" for s in sids[ntr:ntr + nva]} | {s: "test" for s in sids[ntr + nva:]}
    M["part"] = M.sid.map(part)
    tr, va, te = [(M.part == p).to_numpy() for p in ("train", "val", "test")]
    print(f"특징 준비 {time.time() - t0:.0f}s, 테스트 시나리오 {len(sids) - ntr - nva}개", flush=True)

    nte = te.sum()
    hp = a.hide
    conds = {"공지 있음": 0.0, f"일부 공지({a.test_hide:.0%} 숨김)": a.test_hide, "공지 없음": 1.0}  # 테스트 조건 -> 숨김 비율
    Fc = {c: build(g, sci, hide=h)[0][te] for c, h in conds.items()}
    # 그 조건에서 공지된 몰림이 하나라도 있는 시나리오인가 (전환 규칙에 씀)
    ann = {c: M.sid[te].map({s_: len(noisy_plan(sci.loc[s_], hide=h)[1]) > 0 for s_ in M.sid[te].unique()}).to_numpy() for c, h in conds.items()}
    preds = {("직전값", "-"): np.zeros((nte, len(H)))}
    p0_va, p0 = fit_lgbm(F0[tr], Y[tr], F0[va], Y[va], F0[te], list(F0.columns)); preds[("기준(공지 특징 없음)", "-")] = p0
    # 물리 보정: 공지대로면 (1) 출구 용량을 넘는 유입이 쌓이는 양, (2) 통과 중인 인원 변화 = 유입률 변화 × 통과 시간
    cap = F0.cap.to_numpy()
    xs = phys_terms(sci, M, cap)
    coef = [np.clip(np.linalg.lstsq(xs[k][va], (Y[va] - p0_va)[:, k], rcond=None)[0], 0, None) for k in range(len(H))]
    wbest = [[round(float(c), 3) for c in cf] for cf in coef]
    print(f"공지 보정 계수(쌓임, 통과) 5/10/15분 = {wbest}", flush=True)
    for c, h in conds.items():
        xc = xs if h == 0 else phys_terms(sci, M, cap, hide=h)
        preds[("기준 + 공지 보정", c)] = p0 + np.stack([xc[k][te] @ coef[k] for k in range(len(H))], 1)
    names = {"전부 공지 학습": Fa, f"{hp:.0%} 숨김 학습": Fp}
    for name, Ftr in names.items():
        _, p = fit_lgbm(Ftr[tr], Y[tr], Ftr[va], Y[va], pd.concat(Fc.values()), list(Fa.columns))
        for j, c in enumerate(conds):
            preds[(name, c)] = p[j * nte:(j + 1) * nte]
        print(f"{name} 끝 {time.time() - t0:.0f}s", flush=True)
    for name in names:  # 전환: 공지된 몰림이 있으면 그 모델, 하나도 없으면 기준 모델
        for c in conds:
            preds[(f"전환: {name} / 공지 없으면 기준", c)] = np.where(ann[c][:, None], preds[(name, c)], p0)
    Yte, n0, area = Y[te], N0[te], F0.area.to_numpy()[te]
    base = np.abs(Yte).mean(0)
    acc = []
    for (m, c), p in preds.items():
        e = np.abs(p - Yte).mean(0)
        acc.append({"모델": m, "테스트 조건": c, **{f"MAE {h // 60}분": f"{e[k]:.1f} ({1 - e[k] / base[k]:+.1%})" for k, h in enumerate(H)}})
    acc_t = pd.DataFrame(acc)

    Mt = M[te].reset_index(drop=True)
    cur = n0 / area
    al_rows = []
    for thr in THRS:
        ev_by_sid = {}
        for sid, d in g[g.sid.isin(Mt.sid.unique())].groupby("sid"):
            d = d.sort_values("b"); ev_by_sid[sid] = events(d.density.to_numpy(), d.b.to_numpy() * STEP + STEP, thr)
        cands = {("지금 밀도 경보(예측 없음)", "-"): cur >= thr}
        cands |= {k: ((n0[:, None] + p) / area[:, None]).max(1) >= thr for k, p in preds.items() if k[0] != "직전값"}
        for (m, c), al in cands.items():
            n_ev, caught, leads, far = alarm_stats(al, ev_by_sid, cur >= thr, Mt)
            al_rows.append({"기준(명/㎡)": thr, "모델": m, "테스트 조건": c, "사건": n_ev, "잡음": caught,
                            "선행 중앙값(분)": f"{np.median(leads):.1f}" if leads else "-", "선행 범위(분)": f"{min(leads):.1f}~{max(leads):.1f}" if leads else "-",
                            "헛경보 비율": f"{far:.1%}"})
    al_t = pd.DataFrame(al_rows)

    acc_t.to_csv(run / "accuracy.csv", index=False, encoding="utf-8-sig")
    al_t.to_csv(run / "alarm.csv", index=False, encoding="utf-8-sig")
    (run / "config.json").write_text(json.dumps(dict(out=a.out, script="sim_announce", hide_train=a.hide, hide_test=a.test_hide, shapes=a.shapes, sched_noise=dict(time_s=180, size=0.3),
                                                     thresholds=THRS, window_s=WIN, correction_w=wbest, model="LightGBM", n_scenarios=len(sids),
                                                     created=time.strftime("%Y-%m-%d %H:%M")), ensure_ascii=False, indent=1), encoding="utf-8")
    md = "\n\n".join([f"# 공지 있음/없음 테스트 ({a.out})\n\n형태 {a.shapes}, 시나리오 {len(sids)}개, 테스트 {len(sids) - ntr - nva}개(sim_models와 같은 분할), LightGBM. MAE 단위 명, 괄호 = 직전값 대비 개선율.",
                      "## 정확도\n\n" + md_table(acc_t), "## 경보 (사건 = 기준 밀도 20초 이상, 사건 전 15분 안 경보 = 잡음)\n\n" + md_table(al_t)])
    (run / "results.md").write_text(md, encoding="utf-8")
    print(md); print(f"\n전체 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
