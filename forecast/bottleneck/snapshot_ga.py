"""위에서 내려다본 위치 비교: 실제 궤적 vs 시뮬레이션(파라미터 세트별)을 같은 시각에 나란히. 문 앞 구역(0≤y<2m)과 그 순간 밀도 표시.
실행(data-area 루트): C:/Users/Woori/venvs/nf/Scripts/python.exe -m forecast.bottleneck.snapshot_ga [회차 ...] [--sets ga_best.json ga_best_v2.json]
출력: data/crowdqueue/ga_snapshot_<회차>.png"""
import argparse, json
import numpy as np, shapely
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from .calibrate_ga import DATA, FPS, FRONT_DEPTH, CAUTION, GENES, DEFAULT, load_h0_runs, real_metrics, simulate

plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False


def draw(ax, run, pts, title, dens):
    area = shapely.from_wkt(run["wkt"])
    for hole in area.interiors:
        x, y = hole.xy; ax.fill(x, y, color="0.75", lw=0)
    w = run["w"]
    ax.add_patch(plt.Rectangle((-w / 2, 0), w, FRONT_DEPTH, fc=("#ffcccc" if dens >= CAUTION else "#e8f0ff"), ec="tab:red", lw=1))
    ax.scatter(pts[:, 0], pts[:, 1], s=28, c="k", zorder=3)
    ax.set_xlim(-max(w / 2 + 0.6, 1.5), max(w / 2 + 0.6, 1.5)); ax.set_ylim(-2.0, 7.0); ax.set_aspect("equal")
    ax.set_title(f"{title}\n문 앞 {dens:.1f}명/㎡", fontsize=9); ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="*", default=["270_c_34_h0", "090_c_12_h0"])
    ap.add_argument("--sets", nargs="*", default=["ga_best.json", "ga_best_v2.json"]); ap.add_argument("--times", type=int, nargs="*")
    a = ap.parse_args()
    runs = {r["name"]: r for r in load_h0_runs()}
    sets = [("기본값", DEFAULT)] + [(f"GA {f.split('_')[-1].replace('.json', '') if f != 'ga_best.json' else 'v1'}",
                                   np.array([json.loads((DATA / f).read_text(encoding="utf-8"))["best"].get(k, GENES[k][2]) for k in GENES]))
                                  for f in a.sets if (DATA / f).exists()]
    for name in a.runs:
        r = runs[name]; rd, _ = real_metrics(r)
        peak = int(np.argmax(rd)); times = a.times or [max(peak - 8, 1), peak, peak + 12]
        d = r["traj"]; real_at = lambda s: d[d.fr == s * FPS][["x", "y"]].to_numpy()
        sims = []
        for lab, g in sets:
            pos, n = simulate(r, g)
            sims.append((lab, pos))
        fig, axs = plt.subplots(len(times), 1 + len(sims), figsize=(2.6 * (1 + len(sims)), 4.2 * len(times)), squeeze=False)
        area_m2 = r["w"] * FRONT_DEPTH
        for i, s in enumerate(times):
            p = real_at(s)
            draw(axs[i, 0], r, p, f"실제 {s}초", ((p[:, 1] >= 0) & (p[:, 1] < FRONT_DEPTH) & (np.abs(p[:, 0]) <= r["w"] / 2)).sum() / area_m2)
            for j, (lab, pos) in enumerate(sims):
                q = pos[pos.sec == s][["x", "y"]].to_numpy()
                dn = ((q[:, 1] >= 0) & (q[:, 1] < FRONT_DEPTH) & (np.abs(q[:, 0]) <= r["w"] / 2)).sum() / area_m2 if len(q) else 0.0
                draw(axs[i, j + 1], r, q if len(q) else np.zeros((0, 2)), f"{lab} {s}초", dn)
        fig.suptitle(f"{name} (통로 폭 {r['w']}m, {d.id.nunique()}명) — 빨간 테두리 = 문 앞 구역(0~2m), 분홍 = 3명/㎡ 이상", fontsize=10)
        fig.tight_layout(); out = DATA / f"ga_snapshot_{name}.png"; fig.savefig(out, dpi=90); plt.close(fig)
        print("저장:", out, "시각", times)


if __name__ == "__main__":
    main()
