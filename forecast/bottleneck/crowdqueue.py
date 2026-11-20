"""병목 단기 예측 테스트 — 위리히 보행자 실험 'Crowds in front of bottlenecks'(Wuppertal 2018, CC BY 4.0,
DOI 10.34735/ped.2018.1) 궤적으로 '문 앞 구역'의 5·10초 뒤 밀도와 병목 여부를 예측한다.

병목 정의(행안부 다중운집인파사고 안전관리 가이드라인 2024.9 밀도 기준 + 지속 시간):
  주의 = 3명/㎡ 이상, 위험 = 5명/㎡ 이상, 병목(t) = t부터 20초 동안 계속 주의 이상.
구역(미터): 문(폭 0.5m, y<0) 앞 통로 y∈[0,2) = '문 앞', y≥2 = '뒤쪽(상류)'. 통로 폭 w는 회차 메타데이터.
비교: 직전값, 대기열 모형(남은 인원 − 최근 통과 속도×시간), LightGBM. 평가는 회차 하나씩 빼고 학습(24겹).

python -m forecast.bottleneck.crowdqueue      (data/crowdqueue/ 에 trajectories_txt.zip 풀어 둔 상태)
"""
import json, warnings
from pathlib import Path

import lightgbm as lgb, numpy as np, pandas as pd

DATA = Path("data/crowdqueue")
CAUTION, DANGER, HOLD_S = 3.0, 5.0, 20  # 명/㎡, 명/㎡, 초
FRONT_DEPTH = 2.0  # 문 앞 구역 깊이(m)
HORIZONS = (5, 10)
FPS = 25


def load_runs(root=DATA):
    meta = {r["run_name"]: r for r in json.load(open(root / "2018crowdqueue.json", encoding="utf-8"))["experiment"]["run"]}
    for f in sorted((root / "txt").glob("[0-9]*.txt")):
        d = pd.read_csv(f, sep=r"\s+", comment="#", header=None, names=["id", "fr", "x", "y", "z"])
        yield f.stem, d, float(meta[f.stem]["parameter"]["corridor_width"])


def zone_seconds(d, w):
    """궤적 → 1초 단위 구역 지표(팀 영상 파이프라인의 구역 시계열과 같은 성격). 각 초의 첫 프레임 위치 기준."""
    p = d[d.fr % FPS == 0].assign(sec=lambda x: x.fr // FPS).sort_values(["id", "sec"])
    p["speed"] = np.hypot(p.groupby("id").x.diff(), p.groupby("id").y.diff())  # m/s (1초 간격)
    inside = p.x.abs() <= w / 2
    p["zone"] = np.select([(p.y >= 0) & (p.y < FRONT_DEPTH) & inside, (p.y >= FRONT_DEPTH) & inside], ["front", "back"], "out")
    secs = range(p.sec.max() + 1)
    cnt = p.groupby(["sec", "zone"]).id.nunique().unstack(fill_value=0).reindex(secs, fill_value=0)
    spd = p.groupby(["sec", "zone"]).speed.mean().unstack().reindex(secs)
    # 유출 = 문 통과(처음 y<0 된 초), 유입 = 뒤쪽→문 앞 첫 진입 초
    passed = p[p.y < 0].groupby("id").sec.min().value_counts().reindex(secs, fill_value=0)
    entered = p[p.zone == "front"].groupby("id").sec.min().value_counts().reindex(secs, fill_value=0)
    area = w * FRONT_DEPTH
    return pd.DataFrame({"sec": list(secs), "n_front": cnt.get("front", 0).to_numpy(), "n_back": cnt.get("back", 0).to_numpy(),
                         "dens": cnt.get("front", 0).to_numpy() / area, "outflow": passed.to_numpy(), "inflow": entered.to_numpy(),
                         "v_front": spd.get("front", pd.Series(np.nan, index=secs)).to_numpy(),
                         "v_back": spd.get("back", pd.Series(np.nan, index=secs)).to_numpy(), "w": w, "area": area})


def bottleneck_label(dens, thr=CAUTION, hold=HOLD_S):
    """병목(t) = dens[t..t+hold-1] 모두 thr 이상. 기록이 끝나면(군중이 다 빠짐) 아래로 본다."""
    ok = np.asarray(dens) >= thr
    run = np.zeros(len(ok) + 1, int)
    for i in range(len(ok) - 1, -1, -1):  # 뒤에서부터 연속 길이
        run[i] = run[i + 1] + 1 if ok[i] else 0
    return run[:-1] >= hold


def features(z):
    """시점 t까지의 값만 쓰는 특징(누수 없음)."""
    f = pd.DataFrame({"dens": z.dens, "n_back": z.n_back, "remain": z.n_front + z.n_back, "w": z.w,
                      "v_front": z.v_front, "v_back": z.v_back})
    for k in (1, 3, 5):
        f[f"d_dens_{k}"] = z.dens - z.dens.shift(k)
    f["out5"] = z.outflow.rolling(5, min_periods=1).mean()
    f["in5"] = z.inflow.rolling(5, min_periods=1).mean()
    f["back_d3"] = z.n_back - z.n_back.shift(3)
    return f


def queue_forecast(z, h, default_rate=1.3):
    """대기열 모형: 남은 인원이 최근 5초 평균 통과 속도로 빠진다. 뒤에 사람이 남아 있는 동안 문 앞은 지금까지 최대 인원을 유지."""
    rate = z.outflow.rolling(5, min_periods=1).mean().replace(0, np.nan).fillna(default_rate)
    remain = (z.n_front + z.n_back - rate * h).clip(lower=0)
    cap = z.n_front.cummax()
    return np.minimum(remain, cap) / z.area


X = ["dens", "n_back", "remain", "w", "v_front", "v_back", "d_dens_1", "d_dens_3", "d_dens_5", "out5", "in5", "back_d3"]


def build_dataset():
    """회차·시점·예측거리(h)별 특징·정답·기준선 표. LightGBM(main)과 LSTM(crowdqueue_lstm)이 같은 표본을 쓴다."""
    rows = []
    for name, d, w in load_runs():
        z = zone_seconds(d, w)
        f = features(z)
        lab = bottleneck_label(z.dens)
        for h in HORIZONS:
            g = f.copy()
            g["run"], g["sec"], g["h"] = name, z.sec, h
            g["y_dens"] = z.dens.shift(-h)
            g["y_bn"] = pd.Series(lab).shift(-h)
            g["pers"] = z.dens
            g["queue"] = queue_forecast(z, h)
            # 대기열 모형으로 본 병목 여부: t+h부터 20초 동안 예측 밀도가 계속 3 이상
            qpath = np.stack([queue_forecast(z, h + k) for k in range(HOLD_S)])
            g["queue_bn"] = (qpath >= CAUTION).all(0)
            g["pers_bn"] = z.dens >= CAUTION
            rows.append(g.dropna(subset=["y_dens", "y_bn"]))
    df = pd.concat(rows, ignore_index=True)
    df["y_bn"] = df.y_bn.astype(bool)
    return df


def main():
    warnings.filterwarnings("ignore")
    df = build_dataset()
    prm = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10, verbose=-1)

    out = []
    for h in HORIZONS:
        s = df[df.h == h].copy()
        s["lgb"], s["lgb_p"] = np.nan, np.nan
        for r in s.run.unique():  # 회차 하나씩 빼고 학습
            tr, te = s.run != r, s.run == r
            s.loc[te, "lgb"] = lgb.LGBMRegressor(**prm).fit(s.loc[tr, X], s.loc[tr, "y_dens"]).predict(s.loc[te, X])
            if s.loc[tr, "y_bn"].any():
                s.loc[te, "lgb_p"] = lgb.LGBMClassifier(**prm).fit(s.loc[tr, X], s.loc[tr, "y_bn"]).predict_proba(s.loc[te, X])[:, 1]
        out.append(s)
        hot = s.dens >= 2  # 붐비는 순간만 따로
        print(f"\n[{h}초 뒤 문 앞 밀도] MAE(명/㎡) — 전체 {len(s)}초 / 지금 2명/㎡ 이상인 {hot.sum()}초")
        print(pd.DataFrame({m: {"전체": (s[m] - s.y_dens).abs().mean(), "붐빌 때": (s[m] - s.y_dens)[hot].abs().mean()}
                            for m in ["pers", "queue", "lgb"]}).T.rename(index={"pers": "직전값", "queue": "대기열 모형", "lgb": "LightGBM"}).round(3).to_string())
        print(f"[{h}초 뒤 병목(3명/㎡ 이상 20초 지속)] 양성 {int(s.y_bn.sum())}초 / {len(s)}초")
        res = {}
        for m, pred in (("직전값(지금 3 이상)", s.pers_bn), ("대기열 모형", s.queue_bn), ("LightGBM(확률≥0.5)", s.lgb_p >= 0.5)):
            tp, fp, fn = (pred & s.y_bn).sum(), (pred & ~s.y_bn).sum(), (~pred & s.y_bn).sum()
            res[m] = {"정밀도": tp / max(tp + fp, 1), "재현율": tp / max(tp + fn, 1), "헛경보(초)": int(fp), "놓침(초)": int(fn)}
        print(pd.DataFrame(res).T.round(3).to_string())
    pd.concat(out).to_csv(DATA / "pred_bottleneck.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
