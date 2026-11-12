"""영상 구역 시계열 이상 탐지 (명세서 FR-ANO-01 규칙 기반 경보, FR-ANO-02 Isolation Forest).

입력: 구역 시계열 DataFrame[sec, zone, count, motion] (crowd_observations + flow를 sec, zone으로 합친 것,
      backend/app/services/bottleneck.py와 같은 형식).
출력: 명세서 anomaly_events 형식 DataFrame[job_id, zone_id, event_at, event_sec, end_sec, score, method, features].
      score는 0~1. event_at은 base_time + sec (실제 촬영 시작 시각을 넘겨야 한다).
명세서 IF 특징은 [count_delta, motion, reverse_ratio, dwell]이지만, 저장소 샘플에는 사람별 궤적 전체가 없어
지금은 count_delta, motion 두 개만 쓴다. 나머지 두 특징은 열이 있으면 자동으로 포함한다.
연구용 참고 지표이며 실제 사고 위험을 확정 판정하지 않는다."""
from __future__ import annotations

import json
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest

GROWTH_THR, MOTION_DROP_THR, MIN_DURATION_S, LOOKBACK_S, MIN_INCREASE = 0.30, 0.50, 3, 5, 3.0  # 제안값: 샘플 검증 후 튜닝
WINDOW_S, CONTAMINATION, RANDOM_STATE = 10, 0.05, 0  # 명세서 제안: contamination 0.05
IF_FEATURES = ["count_delta", "motion", "reverse_ratio", "dwell"]


def load_zone_series(crowd_csv, flow_csv) -> pd.DataFrame:
    c = pd.read_csv(crowd_csv)[["sec", "zone", "count"]]
    f = pd.read_csv(flow_csv)[["sec", "zone", "motion"]]
    return c.merge(f, on=["sec", "zone"], how="left").sort_values(["zone", "sec"]).reset_index(drop=True)


def _events(mask, sec, min_len):
    """True가 min_len초 이상 이어진 구간 -> [(시작 위치, 끝 위치)]."""
    out, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if sec[i - 1] - sec[start] + 1 >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def rule_events(ts, job_id="sample", base_time="2026-01-01 00:00:00+09:00", growth_thr=GROWTH_THR,
                motion_drop_thr=MOTION_DROP_THR, min_duration_s=MIN_DURATION_S, lookback_s=LOOKBACK_S, min_increase=MIN_INCREASE) -> pd.DataFrame:
    """FR-ANO-01: 인원 급증(최근 lookback초 전 대비 증가율 >= growth_thr) 또는 정체(이동량이 최근 평균 대비
    motion_drop_thr 이상 감소)가 min_duration_s초 이상 이어지면 이벤트. 과거 값만 쓴다(미래 미사용).
    급증은 증가 인원이 min_increase명 이상일 때만 센다(2명 -> 5명처럼 인원이 적을 때 비율만 튀는 것을 막음)."""
    rows = []
    for zone, g in ts.groupby("zone"):
        g = g.sort_values("sec"); sec, cnt, mot = g.sec.to_numpy(), g["count"].to_numpy(float), g.motion.to_numpy(float)
        prev = pd.Series(cnt).shift(lookback_s).to_numpy()
        growth = np.where(cnt - prev >= min_increase, (cnt - prev) / np.maximum(prev, 1.0), np.nan)
        base_m = pd.Series(mot).shift(1).rolling(lookback_s, min_periods=lookback_s).mean().to_numpy()
        drop = 1 - mot / np.where(base_m > 0, base_m, np.nan)
        for kind, val, thr in (("surge", growth, growth_thr), ("stall", drop, motion_drop_thr)):
            for a, b in _events(np.nan_to_num(val, nan=-np.inf) >= thr, sec, min_duration_s):
                peak = float(np.nanmax(val[a:b + 1]))
                rows.append({"job_id": job_id, "zone_id": zone, "event_sec": int(sec[a]), "end_sec": int(sec[b]),
                             "score": round(min(1.0, peak / (2 * thr)), 3), "method": "rule",  # 임계값의 2배를 1.0으로
                             "features": json.dumps({"kind": kind, "peak": round(peak, 3), "threshold": thr, "duration_s": int(sec[b] - sec[a] + 1)}, ensure_ascii=False)})
    return _finish(rows, base_time)


def window_features(ts, window_s=WINDOW_S) -> pd.DataFrame:
    """구역별 슬라이딩 윈도우(1초 간격) 특징. 윈도우 끝 시각 기준이며 그 시각까지의 값만 쓴다."""
    out = []
    for zone, g in ts.groupby("zone"):
        g = g.sort_values("sec").reset_index(drop=True)
        f = pd.DataFrame({"zone": zone, "sec": g.sec})
        f["count_delta"] = g["count"] - g["count"].shift(window_s)
        f["motion"] = g.motion.rolling(window_s).mean()
        for c in ("reverse_ratio", "dwell"):  # 사람별 궤적에서 만든 열이 있으면 사용
            if c in g:
                f[c] = g[c].rolling(window_s).mean()
        out.append(f.dropna(subset=["count_delta", "motion"]))
    return pd.concat(out, ignore_index=True)


def iforest_scores(ts, window_s=WINDOW_S, contamination=CONTAMINATION, random_state=RANDOM_STATE):
    """FR-ANO-02: 구역별 특징 표준화 -> Isolation Forest -> 점수 0~1 정규화(높을수록 이상). 반환: (윈도우별 점수, 사용한 특징)."""
    f = window_features(ts, window_s)
    feats = [c for c in IF_FEATURES if c in f and f[c].notna().all()]
    z = f.groupby("zone")[feats].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) or 1.0))  # 구역별 표준화
    m = IsolationForest(contamination=contamination, random_state=random_state).fit(z)
    raw = -m.score_samples(z)  # 클수록 이상
    f["score"] = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
    f["is_anomaly"] = m.predict(z) == -1
    for c in feats:
        f["z_" + c] = z[c]
    return f, feats


def iforest_events(ts, job_id="sample", base_time="2026-01-01 00:00:00+09:00", **kw) -> pd.DataFrame:
    f, feats = iforest_scores(ts, **kw)
    rows = [{"job_id": job_id, "zone_id": r.zone, "event_sec": int(r.sec), "end_sec": int(r.sec), "score": round(float(r.score), 3), "method": "iforest",
             "features": json.dumps({c: round(float(getattr(r, "z_" + c)), 3) for c in feats}, ensure_ascii=False)}  # 기여 특징 = 표준화 값
            for r in f[f.is_anomaly].itertuples()]
    return _finish(rows, base_time)


def _finish(rows, base_time):
    cols = ["job_id", "zone_id", "event_at", "event_sec", "end_sec", "score", "method", "features"]
    d = pd.DataFrame(rows, columns=[c for c in cols if c != "event_at"])
    d.insert(2, "event_at", pd.Timestamp(base_time) + pd.to_timedelta(d.event_sec, unit="s"))
    return d[cols].sort_values(["zone_id", "event_sec"]).reset_index(drop=True)


def inject_surge(ts, zone, start_sec, length_s=6, factor=2.0, motion_factor=0.3):
    """인수 테스트용: 한 구역에 인원 급증 + 이동량 감소를 인위로 넣는다."""
    t = ts.copy(); m = (t.zone == zone) & t.sec.between(start_sec, start_sec + length_s - 1)
    t.loc[m, "count"] *= factor; t.loc[m, "motion"] *= motion_factor
    return t


if __name__ == "__main__":
    from pathlib import Path
    F = Path(__file__).resolve().parent
    ts = load_zone_series(F / "video_demo_crowd_observations.csv", F / "video_demo_flow.csv")
    calm_r, calm_i = rule_events(ts), iforest_events(ts)
    print(f"[원본 샘플] 규칙 이벤트 {len(calm_r)}건, IF 이상 윈도우 {len(calm_i)}건 (오탐 후보 기록)")
    print(calm_r.drop(columns=["job_id", "event_at"]).to_string()) if len(calm_r) else None
    for fac, mf in ((2.0, 0.3), (3.0, 0.2)):
        inj = inject_surge(ts, "entrance", 40, factor=fac, motion_factor=mf)
        r, (f, feats) = rule_events(inj), iforest_scores(inj)
        hit_rule = ((r.zone_id == "entrance") & r.event_sec.between(40, 45) & r.features.str.contains("surge")).any()
        fe = f[f.zone == "entrance"]; rank = int((fe.score > fe[fe.sec.between(40, 49)].score.max()).sum()) + 1
        print(f"[급증 주입: entrance 40~45초, 인원 {fac}배·이동량 {mf}배] 규칙 감지 {hit_rule} | IF 특징 {feats} | 주입 구간 최고 점수의 entrance 내 순위 {rank}/{len(fe)} | IF 이상 표시 {bool(fe[fe.sec.between(40, 49)].is_anomaly.any())}")
    s1, s2 = iforest_scores(inj)[0].score.to_numpy(), iforest_scores(inj)[0].score.to_numpy()
    print("random_state 고정 시 같은 점수:", np.array_equal(s1, s2))
    print("임계값 변경 반영(증가율 0.3 -> 0.9):", len(rule_events(inj)), "->", len(rule_events(inj, growth_thr=0.9)), "건")
