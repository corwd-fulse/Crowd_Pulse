"""이상 흐름 3단계(낮음/주의/높음) 판정과 사람별 궤적 특징.

- trajectory_features(): 사람별 방향·속도(individual_directions.csv) + 위치(*_detections.csv) -> 구역별 매초 reverse_ratio, dwell
  (명세서 MDL-08 Isolation Forest 특징 4개 중 뒤의 둘. ts에 붙이면 anomaly_video.iforest_scores가 자동으로 쓴다)
- grade_levels(): 규칙 경보 + Isolation Forest + 팀 병목 점수(FR-ZON-05, 40·70 기준)를 합쳐 구역별 매초 단계
- level_events(): '주의' 이상 구간을 이벤트로 묶고 근거 문장·평소 대비 편차(z)를 붙인다(FR-ANO-03 기여 특징 표시용)
연구용 참고 지표이며 실제 상황 판단은 현장 확인이 필요하다. 단계 기준값은 제안값이고 라벨 구간이 생기면 튜닝한다."""
from __future__ import annotations

import json
import numpy as np, pandas as pd

from .anomaly_video import LOOKBACK_S, _events, _finish

# 속도 단위: 초당 몸 높이 배수(카메라 해상도·fps와 무관하게 비교하려고 박스 높이로 나눈다)
MOVE_HPS, SLOW_HPS, DIR_WINDOW_S, TRACK_WINDOW_FRAMES = 0.2, 0.1, 10, 5
LEVELS = ("낮음", "주의", "높음")
BN_WARN, BN_HIGH, IF_WARN = 40.0, 70.0, 0.8  # 병목 점수 기준은 팀 FR-ZON-05와 같다
UP_S, DOWN_S, LONG_WARN_S = 3, 5, 20         # 올릴 때 3초 유지, 내릴 때 5초 풀림, 주의 20초 연속이면 높음


def _zone_name(z):
    return z["name"] if isinstance(z, dict) else z.name


def _zone_poly(z):
    return np.array(z["polygon"] if isinstance(z, dict) else z.polygon, np.float32)


def _in_poly(pts, poly):
    """점들이 다각형 안(경계 포함)인지 = cv2.pointPolygonTest(...) >= 0 과 같은 판정. numpy만 써서 예측 환경에 opencv가 없어도 된다."""
    p = np.asarray(pts, float).reshape(-1, 2); x, y = p[:, :1], p[:, 1:]
    a = np.asarray(poly, float); b = np.roll(a, -1, 0)
    ax, ay, dx, dy = a[:, 0], a[:, 1], b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    crosses = ((ay > y) != (b[:, 1] > y)) & (x < dx * (y - ay) / np.where(dy == 0, 1, dy) + ax)  # 오른쪽 반직선과 만나는 변 수
    t = np.clip(((x - ax) * dx + (y - ay) * dy) / np.maximum(dx * dx + dy * dy, 1e-12), 0, 1)
    on_edge = (np.hypot(ax + t * dx - x, ay + t * dy - y) < 1e-6).any(1)
    return (crosses.sum(1) % 2 == 1) | on_edge


def trajectory_features(directions_csv, detections_csv, zones, fps, window_frames=TRACK_WINDOW_FRAMES):
    """-> DataFrame[sec, zone, reverse_ratio, dwell, n_people].
    구역 배정: 박스 하단 중앙이 구역 다각형 안(팀 zone_analysis와 같은 기준).
    속도 v = speed_px / window_frames * fps / 박스 높이. dwell = 구역 안 사람 중 v < SLOW_HPS 비율.
    reverse_ratio = 움직이는(v > MOVE_HPS) 사람 중, 최근 DIR_WINDOW_S초 구역 주 방향(원형 평균)과 90도 넘게 다른 비율."""
    ind, det = pd.read_csv(directions_csv), pd.read_csv(detections_csv)
    m = ind[ind.reliable.astype(bool)].merge(det, on=["frame", "track_id"], suffixes=("", "_d"))
    bx, by = (m.x1 + m.x2) / 2, m.y2
    pts, zone = np.column_stack([bx, by]), np.full(len(m), None, dtype=object)
    for z in reversed(zones):  # 겹치면 앞 구역 우선(이전 next(...)와 같음)
        zone[_in_poly(pts, _zone_poly(z))] = _zone_name(z)
    m["zone"] = zone
    m = m.dropna(subset=["zone"]).copy()
    m["sec"] = (m.frame / fps).astype(int)
    m["v"] = m.speed_px / window_frames * fps / (m.y2 - m.y1).clip(lower=1)
    rows = []
    for z, g in m.groupby("zone"):
        secs = range(int(g.sec.min()), int(g.sec.max()) + 1)
        mov = g[(g.v > MOVE_HPS) & g.angle_deg.notna()]
        rad = np.deg2rad(mov.angle_deg.to_numpy())
        per = pd.DataFrame({"sec": mov.sec.to_numpy(), "c": np.cos(rad), "s": np.sin(rad)}).groupby("sec")[["c", "s"]].sum()
        per = per.reindex(secs, fill_value=0).rolling(DIR_WINDOW_S, min_periods=1).sum()  # 지금까지 10초(과거만)
        main = (np.rad2deg(np.arctan2(per.s, per.c)) % 360).rename("main")
        diff = ((mov.angle_deg - mov.sec.map(main) + 180) % 360 - 180).abs()
        rev = (diff > 90).groupby(mov.sec).mean()
        ps = g.groupby("sec").agg(n_people=("track_id", "nunique"), dwell=("v", lambda v: float((v < SLOW_HPS).mean())))
        ps["reverse_ratio"] = rev.reindex(ps.index).fillna(0.0)
        rows.append(ps.reindex(secs).reset_index().rename(columns={"index": "sec"}).assign(zone=z))
    return pd.concat(rows, ignore_index=True)[["sec", "zone", "reverse_ratio", "dwell", "n_people"]]


ARRIVAL_HS = (10, 30, 60)


def _read_det(detections):
    det = detections.copy() if isinstance(detections, pd.DataFrame) else pd.read_csv(detections)
    return det.dropna(subset=["track_id"]).sort_values(["track_id", "t_ms"]).reset_index(drop=True)


def stitch_tracks(detections, max_gap_s=1.0, max_dist_h=1.0):
    """끊긴 추적 잇기: 끝난 ID a와 max_gap_s초 안에 새로 생긴 ID b를, a의 마지막 위치를 a 속도로 공백만큼 민 점과 b의 첫 위치가
    키(박스 높이) × max_dist_h 이내면 같은 사람으로 본다(가까운 쌍부터, a 하나에 b 하나). -> track_id를 처음 ID로 바꾼 표(raw_track_id 보관)."""
    det = _read_det(detections)
    det["t"] = det.t_ms / 1000.0
    det["fx"], det["fy"] = (det.x1 + det.x2) / 2, det.y2
    g = det.groupby("track_id")
    first, last = g.first(), g.last()
    back = det[det.t >= det.track_id.map(last.t) - 1.0].groupby("track_id").first()  # 끝나기 1초 전 관측(속도용)
    dt = (last.t - back.t).where(lambda s: s > 0.2)
    vx, vy = ((last.fx - back.fx) / dt).fillna(0), ((last.fy - back.fy) / dt).fillna(0)
    cand = []
    starts = first.sort_values("t")
    for b, rb in starts.iterrows():
        ends = last[(last.t < rb.t) & (last.t >= rb.t - max_gap_s)]
        if ends.empty:
            continue
        gap = rb.t - ends.t
        d = np.hypot(ends.fx + vx[ends.index] * gap - rb.fx, ends.fy + vy[ends.index] * gap - rb.fy)
        lim = max_dist_h * ((ends.y2 - ends.y1) + (rb.y2 - rb.y1)) / 2
        for a in d[d <= lim].index:
            cand.append((float(d[a]), a, b))
    parent, used_a, used_b = {}, set(), set()
    for _, a, b in sorted(cand):
        if a not in used_a and b not in used_b:
            parent[b] = a; used_a.add(a); used_b.add(b)
    root = lambda x: root(parent[x]) if x in parent else x
    det["raw_track_id"] = det.track_id
    det["track_id"] = det.track_id.map(lambda x: root(x))
    return det.drop(columns=["t", "fx", "fy"]).sort_values(["track_id", "t_ms"]).reset_index(drop=True)


def _track_kinematics(det, hist_s=2.0, min_span_s=1.0):
    """사람(ID)·초마다 그 초 끝까지의 마지막 발 위치와 최근 hist_s초 평균 속도(px/s). known = 이력 min_span_s초 이상, moving = 빠름(MOVE_HPS 이상)."""
    S, ID, Pp, V, Hh, KN = [], [], [], [], [], []
    for tid, g in det.groupby("track_id"):
        tt = g.t_ms.to_numpy() / 1000.0
        pos = np.column_stack([(g.x1 + g.x2).to_numpy() / 2, g.y2.to_numpy()])
        hgt = (g.y2 - g.y1).clip(lower=1).to_numpy()
        secs = np.unique(np.floor(tt)).astype(int)
        i = np.searchsorted(tt, secs + 1, "left") - 1          # 그 초 끝까지의 마지막 관측
        j = np.searchsorted(tt, tt[i] - hist_s, "left")        # hist_s초 전 관측
        span = tt[i] - tt[j]
        known = span >= min_span_s
        v = np.where(known[:, None], (pos[i] - pos[j]) / np.maximum(span, 1e-9)[:, None], 0.0)
        S.append(secs); ID.append(np.full(len(secs), tid)); Pp.append(pos[i]); V.append(v); Hh.append(hgt[i]); KN.append(known)
    K = pd.DataFrame({"sec": np.concatenate(S), "track_id": np.concatenate(ID)})
    P_, V_, H_ = np.vstack(Pp), np.vstack(V), np.concatenate(Hh)
    K["x"], K["y"], K["vx"], K["vy"], K["h"], K["known"] = P_[:, 0], P_[:, 1], V_[:, 0], V_[:, 1], H_, np.concatenate(KN)
    K["moving"] = K.known & (np.hypot(K.vx, K.vy) / K.h >= MOVE_HPS)
    return K


def build_flow_map(detections, until_s=None, cell=60, min_n=5, hist_s=2.0, min_span_s=1.0):
    """흐름 지도: 화면을 cell px 격자로 나눠 칸마다 '여기 있던 움직이는 사람의 평균 속도(px/s)'. until_s초 전 관측만 사용(미래 누수 방지).
    표본 min_n개 미만 칸은 0(모름). -> dict(cell, vx[ny, nx], vy[ny, nx], n[ny, nx])"""
    det = _read_det(detections)
    K = _track_kinematics(det, hist_s, min_span_s)
    K = K[K.moving & ((K.sec < until_s) if until_s is not None else True)]
    W_, H_ = float(det.x2.max()), float(det.y2.max())
    nx, ny = int(W_ // cell) + 1, int(H_ // cell) + 1
    ix, iy = np.clip((K.x // cell).astype(int), 0, nx - 1), np.clip((K.y // cell).astype(int), 0, ny - 1)
    flat = iy * nx + ix
    n = np.bincount(flat, minlength=nx * ny)
    sx, sy = np.bincount(flat, K.vx, nx * ny), np.bincount(flat, K.vy, nx * ny)
    ok = n >= min_n
    vx, vy = np.where(ok, sx / np.maximum(n, 1), 0.0), np.where(ok, sy / np.maximum(n, 1), 0.0)
    return {"cell": cell, "vx": vx.reshape(ny, nx), "vy": vy.reshape(ny, nx), "n": n.reshape(ny, nx)}


def arrival_eta(detections, zones, horizon=max(ARRIVAL_HS), hist_s=2.0, min_span_s=1.0, stitch_gap_s=0.0, flow_map=None, tau_s=10.0):
    """사람·초마다 구역별 지금 안에 있는지(in_<구역>)와 들어갈/나갈 때까지 남은 초(eta_in_<구역>, eta_out_<구역>, horizon초 안에 없으면 inf).
    이동: 직선(최근 속도). flow_map을 주면 k초 뒤 속도 = e^(-k/tau_s)·내 속도 + (1-e^(-k/tau_s))·지도 속도(그 위치),
    이력이 짧아 속도를 모르는 사람은 지도 속도만 쓴다. 이력이 충분한데 느린 사람은 제자리."""
    det = _read_det(detections)
    if stitch_gap_s > 0:
        det = stitch_tracks(det, stitch_gap_s)
    K = _track_kinematics(det, hist_s, min_span_s)
    P0 = K[["x", "y"]].to_numpy(float)
    V0 = np.where(K.moving.to_numpy()[:, None], K[["vx", "vy"]].to_numpy(float), 0.0)
    use_map = flow_map is not None
    if use_map:
        fm, c = flow_map, flow_map["cell"]
        ny, nx = fm["vx"].shape
        follow = K.moving.to_numpy() | ~K.known.to_numpy()  # 움직이는 사람 + 속도를 모르는 사람만 지도를 따른다
    paths = [(_zone_name(z), _zone_poly(z)) for z in zones]
    now = {nm: _in_poly(P0, p) for nm, p in paths}
    eta_in = {nm: np.full(len(K), np.inf) for nm, _ in paths}; eta_out = {nm: np.full(len(K), np.inf) for nm, _ in paths}
    pos = P0.copy()
    for k in range(1, horizon + 1):
        if use_map:
            ix, iy = np.clip((pos[:, 0] // c).astype(int), 0, nx - 1), np.clip((pos[:, 1] // c).astype(int), 0, ny - 1)
            vm = np.column_stack([fm["vx"][iy, ix], fm["vy"][iy, ix]])
            w = np.where(K.known.to_numpy(), np.exp(-(k - 0.5) / tau_s), 0.0)[:, None]
            pos = pos + np.where(follow[:, None], w * V0 + (1 - w) * vm, 0.0)
        else:
            pos = P0 + V0 * k
        for nm, p in paths:
            ins = _in_poly(pos, p)
            eta_in[nm] = np.where(np.isinf(eta_in[nm]) & ~now[nm] & ins, k, eta_in[nm])
            eta_out[nm] = np.where(np.isinf(eta_out[nm]) & now[nm] & ~ins, k, eta_out[nm])
    for nm, _ in paths:
        K[f"in_{nm}"], K[f"eta_in_{nm}"], K[f"eta_out_{nm}"] = now[nm], eta_in[nm], eta_out[nm]
    return K, det


def arrival_features(detections, zones, horizons=ARRIVAL_HS, hist_s=2.0, min_span_s=1.0, stitch_gap_s=0.0, flow_map=None,
                     birth_windows=(60, 120)):
    """도착·이탈 예정 인원 -> DataFrame[sec, zone, arr{H}, dep{H}, births_in{W}, births_out{W}] (예측 특징, 과거 관측만 사용).
    arrH = 지금 구역 밖인데 H초 안에 들어갈 사람 수, depH = 지금 안인데 H초 안에 나갈 사람 수 (이동 방식은 arrival_eta).
    births_*{W} = 최근 W초 동안 새로 나타난 사람(ID) 수, 구역 안/밖에서 처음 보인 것으로 나눔 — 아직 화면에 없는 사람의 유입률 대용.
    stitch_gap_s > 0 이면 끊긴 추적을 먼저 잇는다(stitch_tracks)."""
    K, det = arrival_eta(detections, zones, max(horizons), hist_s, min_span_s, stitch_gap_s, flow_map)
    secs_all = np.arange(0, int(K.sec.max()) + 1)
    born = K.sort_values("sec").groupby("track_id").head(1)
    rows = []
    for z in zones:
        nm = _zone_name(z)
        now, ei, eo = K[f"in_{nm}"].to_numpy(), K[f"eta_in_{nm}"].to_numpy(), K[f"eta_out_{nm}"].to_numpy()
        f = pd.DataFrame({"sec": K.sec, **{f"arr{h}": (~now) & (ei <= h) for h in horizons}, **{f"dep{h}": now & (eo <= h) for h in horizons}})
        f = f.groupby("sec").sum().reindex(secs_all, fill_value=0)
        for side, msk in (("in", born[f"in_{nm}"]), ("out", ~born[f"in_{nm}"])):
            b = born[msk].groupby("sec").size().reindex(secs_all, fill_value=0)
            for wnd in birth_windows:
                f[f"births_{side}{wnd}"] = b.rolling(wnd, min_periods=1).sum().to_numpy()
        rows.append(f.rename_axis("sec").reset_index().assign(zone=nm))
    return pd.concat(rows, ignore_index=True)


def hysteresis(cand, up_s=UP_S, down_s=DOWN_S, long_s=LONG_WARN_S):
    """후보 단계 열(0/1/2) -> 흔들림을 줄인 단계 열. up_s초 이상 유지돼야 올리고, down_s초 이상 풀려야 내린다.
    '주의'가 long_s초 이상 이어지면 '높음'으로 올린다."""
    lvl, out, run_up, run_dn, run_warn = 0, [], {1: 0, 2: 0}, 0, 0
    for c in cand:
        for k in (1, 2):
            run_up[k] = run_up[k] + 1 if c >= k else 0
        if c > lvl:
            run_dn = 0
            for k in (2, 1):
                if k > lvl and run_up[k] >= up_s:
                    lvl = k
                    break
        elif c < lvl:
            run_dn += 1
            if run_dn >= down_s:
                lvl, run_dn = int(c), 0
        else:
            run_dn = 0
        run_warn = run_warn + 1 if lvl >= 1 else 0
        if lvl == 1 and run_warn >= long_s:
            lvl = 2
        out.append(lvl)
    return np.array(out, dtype=int)


def grade_levels(ts, rules=None, if_scores=None, bottleneck=None, bn_window_s=10):
    """구역별 매초 단계 타임라인 -> DataFrame[zone_id, sec, level, level_num, candidate, 신호들...].
    ts: [sec, zone, count, motion(, reverse_ratio, dwell)], rules: rule_events() 결과, if_scores: iforest_scores()[0],
    bottleneck: 팀 bottleneck_scores.csv(window_start, zone, score). 없는 신호는 빼고 판정한다."""
    out = []
    for z, g in ts.groupby("zone"):
        g = g.sort_values("sec").reset_index(drop=True)
        t = pd.DataFrame({"zone_id": z, "sec": g.sec, "count": g["count"]})
        t["count_delta_5s"] = (g["count"] - g["count"].shift(LOOKBACK_S)).round(2)
        base = g.motion.shift(1).rolling(LOOKBACK_S, min_periods=LOOKBACK_S).mean()
        t["motion_drop"] = (1 - g.motion / base.where(base > 0)).round(3)
        for c in ("reverse_ratio", "dwell"):
            t[c] = g[c].round(3) if c in g else np.nan
        t["bottleneck_score"] = np.nan
        if bottleneck is not None and len(bottleneck):
            b = bottleneck[bottleneck.zone == z].set_index("window_start").score
            t["bottleneck_score"] = (t.sec // bn_window_s * bn_window_s).map(b)
        t["rule_kind"], t["rule_strength"] = "", 0.0
        if rules is not None and len(rules):
            for r in rules[rules.zone_id == z].itertuples():
                k = json.loads(r.features).get("kind", "")
                sel = t.sec.between(r.event_sec, r.end_sec)
                t.loc[sel, "rule_kind"] = [k if not cur else (cur if k in cur else cur + "+" + k) for cur in t.loc[sel, "rule_kind"]]
                t.loc[sel, "rule_strength"] = np.maximum(t.loc[sel, "rule_strength"], r.score)
        t["if_score"], t["if_anomaly"] = np.nan, False
        if if_scores is not None and len(if_scores):
            f = if_scores[if_scores.zone == z].set_index("sec")
            t["if_score"] = t.sec.map(f.score).round(3)
            t["if_anomaly"] = t.sec.map(f.is_anomaly).fillna(False).astype(bool)
        rule_on, bn = t.rule_kind != "", t.bottleneck_score.fillna(0)
        high = (bn >= BN_HIGH) | (rule_on & t.if_anomaly)
        warn = (bn >= BN_WARN) | rule_on | (t.if_score.fillna(0) >= IF_WARN)
        t["candidate"] = np.where(high, 2, np.where(warn, 1, 0))
        t["level_num"] = hysteresis(t.candidate.to_numpy())
        t["level"] = t.level_num.map(dict(enumerate(LEVELS)))
        out.append(t)
    cols = ["zone_id", "sec", "level", "level_num", "candidate", "count", "count_delta_5s", "motion_drop", "reverse_ratio", "dwell",
            "rule_kind", "rule_strength", "if_score", "if_anomaly", "bottleneck_score"]
    return pd.concat(out, ignore_index=True)[cols]


def reason_text(r, cnt_prev=np.nan, mu=None, sd=None):
    """근거 문장: 기준을 넘었거나 눈에 띄는 신호만 나열한다. 역방향·정체는 그 구역 평소(mu, sd)보다 눈에 띄게 높을 때만."""
    hi = lambda f, floor: pd.notna(r[f]) and (mu is None or r[f] - mu.get(f, 0) >= max(floor, sd.get(f, 0) if sd is not None and pd.notna(sd.get(f)) else 0))
    parts = []
    if pd.notna(r.count_delta_5s) and abs(r.count_delta_5s) >= 1:
        pct = f"({r.count_delta_5s / cnt_prev * 100:+.0f}%)" if pd.notna(cnt_prev) and cnt_prev > 0 else ""
        parts.append(f"5초 전 대비 인원 {r.count_delta_5s:+.1f}명{pct}")
    if pd.notna(r.motion_drop) and r.motion_drop >= 0.3:
        parts.append(f"이동량 -{r.motion_drop * 100:.0f}%")
    if hi("reverse_ratio", 0.15):
        parts.append(f"역방향 {r.reverse_ratio * 100:.0f}%(평소 {mu['reverse_ratio'] * 100:.0f}%)" if mu is not None else f"역방향 {r.reverse_ratio * 100:.0f}%")
    if hi("dwell", 0.15):
        parts.append(f"거의 서 있는 사람 {r.dwell * 100:.0f}%(평소 {mu['dwell'] * 100:.0f}%)" if mu is not None else f"거의 서 있는 사람 {r.dwell * 100:.0f}%")
    if r.rule_kind:
        parts.append(" + ".join({"surge": "급증 경보", "stall": "정체 경보"}.get(k, k) for k in r.rule_kind.split("+")))
    if pd.notna(r.if_score) and r.if_score >= IF_WARN:
        parts.append(f"IF {r.if_score:.2f}")
    if pd.notna(r.bottleneck_score) and r.bottleneck_score >= BN_WARN:
        parts.append(f"병목 {r.bottleneck_score:.0f}")
    return " · ".join(parts) or "개별 기준 미만의 복합 신호"


EVENT_COLUMNS = ["job_id", "zone_id", "event_at", "event_sec", "end_sec", "score", "method", "features", "level", "duration_s", "peak_sec", "reason"]
Z_FEATURES = ["count_delta_5s", "motion_drop", "reverse_ratio", "dwell", "if_score", "bottleneck_score"]


def level_events(timeline, job_id="sample", base_time="2026-01-01 00:00:00+09:00"):
    """'주의' 이상이 이어진 구간 -> 이벤트(method='level'). 최고 단계, 지속 시간, 최고 시각, 근거 문장, 평소 대비 편차 z(features)."""
    rows = []
    for z, t in timeline.groupby("zone_id"):
        t = t.sort_values("sec").reset_index(drop=True)
        cnt = t.set_index("sec")["count"]
        mu, sd = t[Z_FEATURES].mean(), t[Z_FEATURES].std(ddof=0).replace(0, np.nan)
        for a, b in _events(t.level_num.to_numpy() >= 1, t.sec.to_numpy(), 1):
            seg = t.iloc[a:b + 1]
            pk = seg.sort_values(["level_num", "candidate", "rule_strength", "if_score"], ascending=False).iloc[0]
            zv = {f: round(float((pk[f] - mu[f]) / sd[f]), 2) for f in Z_FEATURES if pd.notna(pk[f]) and pd.notna(sd[f])}
            rows.append({"job_id": job_id, "zone_id": z, "event_sec": int(seg.sec.iloc[0]), "end_sec": int(seg.sec.iloc[-1]),
                         "score": round(float(seg.level_num.max() / 2), 3), "method": "level", "features": json.dumps(zv, ensure_ascii=False),
                         "level": LEVELS[int(seg.level_num.max())], "duration_s": int(seg.sec.iloc[-1] - seg.sec.iloc[0] + 1),
                         "peak_sec": int(pk.sec), "reason": reason_text(pk, cnt.get(int(pk.sec) - LOOKBACK_S, np.nan), mu, sd)})
    base = _finish([{k: r[k] for k in ("job_id", "zone_id", "event_sec", "end_sec", "score", "method", "features")} for r in rows], base_time)
    extra = pd.DataFrame(rows, columns=["zone_id", "event_sec", "level", "duration_s", "peak_sec", "reason"])
    return base.merge(extra, on=["zone_id", "event_sec"], how="left")[EVENT_COLUMNS]
