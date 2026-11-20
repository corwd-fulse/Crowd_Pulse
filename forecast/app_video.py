"""영상 → 추출 CSV → 예측·이상 흐름 결과를 한 화면에서 보는 뷰어.
실행: VIDEO_RUNS=runs.json streamlit run forecast/app_video.py
runs.json: [{"name": "...", "video": "원본 영상", "detections": "*_detections.csv", "zones_dir": "run_zone_pipeline 결과 폴더(+ forecast_video.csv, anomaly_events.csv)",
             "zones": [{"name": "inner", "polygon": [[x, y], ...]}, ...]}]"""
import json, os, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 저장소 루트(forecast 패키지)

st.set_page_config(page_title="영상 분석 → 예측 · 이상 흐름", layout="wide")
st.title("영상 분석 → 구역 인원 예측 · 이상 흐름 탐지")
st.warning("참고 지표입니다. 이상 점수와 경보는 운영자의 확인을 돕는 용도이며, 실제 상황 판단은 현장 확인이 필요합니다.")

runs = json.loads(Path(os.environ.get("VIDEO_RUNS", "runs.json")).read_text(encoding="utf-8"))
run = runs[[r["name"] for r in runs].index(st.sidebar.selectbox("영상", [r["name"] for r in runs]))]
COLORS = [(255, 170, 0), (0, 200, 255), (120, 255, 120), (255, 80, 200)]  # BGR, 구역별


@st.cache_data
def load(run_json, mtime):  # mtime: 결과 CSV가 다시 만들어지면 캐시를 새로 읽게
    r = json.loads(run_json); z = Path(r["zones_dir"])
    det = pd.read_csv(r["detections"])
    obs = pd.read_csv(z / "crowd_observations.csv")
    fc = pd.read_csv(z / "forecast_video.csv", encoding="utf-8-sig") if (z / "forecast_video.csv").exists() else pd.DataFrame()
    an = pd.read_csv(z / "anomaly_events.csv", encoding="utf-8-sig") if (z / "anomaly_events.csv").exists() else pd.DataFrame()
    bn = pd.read_csv(z / "bottleneck_scores.csv") if (z / "bottleneck_scores.csv").exists() else pd.DataFrame()
    tl = pd.read_csv(z / "anomaly_timeline.csv", encoding="utf-8-sig") if (z / "anomaly_timeline.csv").exists() else pd.DataFrame()
    cap = cv2.VideoCapture(r["video"]); meta = (cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    return det, obs, fc, an, bn, tl, meta


det, obs, fc, an, bn, tl, (fps, nframes) = load(json.dumps(run), max((p.stat().st_mtime for p in Path(run["zones_dir"]).glob("*.csv")), default=0))
LEVEL_BGR = {1: (0, 210, 255), 2: (0, 0, 255)}  # 주의 노랑, 높음 빨강
secs = int(obs.sec.max())
zones = run["zones"]


def frame_at(sec):
    cap = cv2.VideoCapture(run["video"]); f = int(min(sec * fps, nframes - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
    if not ok:
        return None
    over = img.copy()
    for k, z in enumerate(zones):
        pts = np.array(z["polygon"], np.int32)
        cv2.fillPoly(over, [pts], COLORS[k % 4])
    img = cv2.addWeighted(over, 0.18, img, 0.82, 0)
    for k, z in enumerate(zones):
        pts = np.array(z["polygon"], np.int32); cv2.polylines(img, [pts], True, COLORS[k % 4], 2)
        n = obs[(obs.sec == sec) & (obs.zone == z["name"])]["count"]
        lv = tl[(tl.sec == sec) & (tl.zone_id == z["name"])].level_num if len(tl) else pd.Series(dtype=int)
        lvn = int(lv.iloc[0]) if len(lv) else 0
        tag = ["LOW", "WARN", "HIGH"][lvn]  # OpenCV 글꼴이 한글을 못 그려 영문 약어(낮음/주의/높음)
        if lvn:
            cv2.polylines(img, [pts], True, LEVEL_BGR[lvn], 4)
        label = (f"{z['name']}: {n.iloc[0]:.1f}" if len(n) else z["name"]) + f" [{tag}]"
        cv2.putText(img, label, tuple(pts.min(0) + [8, 28]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, LEVEL_BGR.get(lvn, COLORS[k % 4]), 2)
    for r in det[det.frame == f].itertuples():
        cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (60, 220, 60), 1)
        cv2.putText(img, str(r.track_id), (int(r.x1), int(r.y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 220, 60), 1)
    top = int(tl[tl.sec == sec].level_num.max()) if len(tl) and (tl.sec == sec).any() else 0
    if top:
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), LEVEL_BGR[top], 6)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


c = st.sidebar
sec = c.slider("영상 시각(초)", 0, secs, min(45, secs))
play = c.toggle("재생 (1초씩)")
win = c.select_slider("그래프에 보일 구간(초)", [60, 120, 300, 600, secs + 1], value=min(120, secs + 1), format_func=lambda v: "전체" if v == secs + 1 else f"최근 {v}초")
hz = c.radio("예측 시간", sorted(fc.horizon_s.unique()) if len(fc) and "horizon_s" in fc else [10], format_func=lambda h: f"{h}초 뒤", horizontal=True)
avail = [m for m in ["persistence", "moving_avg_5s", "ridge_ar", "lightgbm", "lightgbm_pre", "lstm_pre"]
         if len(fc) and (fc[fc.model == m].status == "ok").any()] or ["persistence"]
sel = c.multiselect("예측 모델", avail, default=[m for m in ["persistence", "lightgbm", "lstm_pre"] if m in avail] or avail[:1],
                    help="persistence·moving_avg는 5초부터, ridge_ar는 약 35초부터, lightgbm은 약 4분 이후부터 예측. *_pre는 미리 학습한 모델(학습 없이 처음부터)")
st.sidebar.caption(f"영상 {nframes}프레임, {fps:.0f}fps, {secs + 1}초 · 구역 {', '.join(z['name'] for z in zones)}")
if "vsec" not in st.session_state or st.session_state.get("_base") != sec:
    st.session_state.vsec, st.session_state["_base"] = sec, sec


@st.fragment(run_every=1.0 if play else None)
def view():
    if play:
        st.session_state.vsec = (st.session_state.vsec + 1) % (secs + 1)
    s = st.session_state.vsec
    L, R = st.columns([3, 2])
    img = frame_at(s)
    if img is not None:  # 조건식 한 줄로 쓰면 Streamlit magic이 반환 객체를 화면에 출력한다
        L.image(img, caption=f"{s}초 — 초록 박스: YOLO+ByteTrack 추적 · 구역 옆 [LOW/WARN/HIGH] = 낮음/주의/높음 · 테두리: 노랑 주의, 빨강 높음", width="stretch")
    else:
        L.error("프레임을 읽지 못했습니다.")
    with R:
        H = int(hz)
        fch = fc[fc.horizon_s == H] if "horizon_s" in fc else fc
        tgt = fch.target.iloc[0] if len(fch) and "target" in fch else "raw"
        lo = max(0, s - win)
        st.markdown(f"**구역별 인원과 {H}초 뒤 예측 (최근 {win}초)** — 선 이름 '실제' = 실제 {'10초 평균' if tgt == 'mean10' else '인원'}, 나머지 = 모델의 {H}초 뒤 예측. "
                    f"예측선은 각 시각의 {H}초 전까지만 보고 낸 값이라 현재 시각보다 {H}초 앞까지 그려진다")
        for z in zones:
            a0 = obs[obs.zone == z["name"]].set_index("sec")["count"]
            a = (a0.rolling(10).mean() if tgt == "mean10" else a0).loc[lo:s].rename("실제")
            q = fch[(fch.zone_id == z["name"]) & fch.model.isin(sel) & (fch.last_observed_sec <= s) & (fch.target_sec >= lo)] if len(fch) else fch
            lines = [a] + [q[q.model == m].set_index("target_sec")["value"].rename(m) for m in sel]
            now = q[q.last_observed_sec == s]
            txt = " · ".join(f"{r.model} {r.value:.1f}명" if r.status == "ok" else f"{r.model} {('학습 데이터 부족' if r.status == 'insufficient_history' else r.status)}" for r in now.itertuples())
            st.caption(f"{z['name']} — 지금({s}초) 보고 낸 {s + H}초 예측: {txt}" if txt else z["name"])
            st.line_chart(pd.concat(lines, axis=1), height=170, x_label="영상 시각(초)")
        if len(fc):
            nxt = fch[(fch.split == "forecast") & fch.model.isin(sel)]
            st.markdown(f"**영상이 끝난 뒤 {H}초 예측**: " + " · ".join(f"{r.zone_id}/{r.model} {r.value:.1f}명" if r.status == "ok" else f"{r.zone_id}/{r.model} 학습 데이터 부족" for r in nxt.itertuples()))


view()

ap = Path(run["zones_dir"]) / "arrival_compare.csv"
if ap.exists():  # 실험 스크립트(arrival_test.py) 결과가 있을 때만
    st.subheader("도착 예정 특징(A) 효과 — 추적 궤적으로 'H초 안에 들어올·나갈 사람 수'를 특징으로 추가")
    ac = pd.read_csv(ap, encoding="utf-8-sig")
    show = [m for m in ac.model.unique() if m != "persistence"]
    sk = ac[ac.model.isin(show)].pivot_table(index="model", columns="horizon_s", values="skill", sort=False)
    sk.columns = [f"{c}초 뒤" for c in sk.columns]
    L2, R2 = st.columns([3, 2])
    L2.bar_chart(sk.mul(100).round(1), stack=False, horizontal=True, height=380, x_label="직전값 대비 개선율(%)", y_label="")
    R2.dataframe(sk.mul(100).round(1).astype(str).add("%"), width="stretch", height=380)
    st.caption("뒤 40%, 같은 시점 비교. +A = 도착·이탈 예정 특징 추가, lightgbm_D = 단순화한 LightGBM 설정, +lstm = LSTM(시드 0)과 평균. "
               "흔들림·MAE·RMSE는 아래 표.")
    st.dataframe(ac.rename(columns={"horizon_s": "예측 시간(초)", "model": "모델", "skill": "직전값 대비 개선율", "jump": "흔들림(jump)", "n": "비교 시점 수"}),
                 hide_index=True, width="stretch", height=250)
    st.markdown("**A를 넣은 효과가 우연인지 (60초 블록 부트스트랩, MAE 차이 95% 구간)**")
    st.dataframe(pd.read_csv(ap.with_name("arrival_bootstrap.csv"), encoding="utf-8-sig").rename(columns={"horizon_s": "예측 시간(초)"}),
                 hide_index=True, width="stretch", height=250)
    afp = ap.with_name("arrival_feats.csv")
    if afp.exists():
        af = pd.read_csv(afp)
        Hs = st.radio("신호 확인: 예측 시간", [10, 30, 60], index=1, horizontal=True, format_func=lambda h: f"{h}초")
        lo = max(0, sec - win)
        st.markdown(f"**순증 예정(도착−이탈, {Hs}초 안) vs 실제로 {Hs}초 뒤 늘어난 인원** — 최근 {win}초, 사이드바 영상 시각 기준. 두 선이 같이 움직이면 신호가 있다는 뜻")
        cols = st.columns(len(zones))
        for col, z in zip(cols, zones):
            y10 = obs[obs.zone == z["name"]].set_index("sec")["count"].rolling(10).mean()
            a_ = af[af.zone == z["name"]].set_index("sec")
            dfz = pd.DataFrame({"순증 예정": a_[f"arr{Hs}"] - a_[f"dep{Hs}"], f"실제 {Hs}초 뒤 변화": y10.shift(-Hs) - y10}).loc[lo:sec]
            col.caption(z["name"]); col.line_chart(dfz, height=200, x_label="영상 시각(초)", y_label="명")

bp = Path(run["zones_dir"]) / "bins10_preds.csv"
if bp.exists():  # 실험 스크립트(bins10_test.py) 결과가 있을 때만
    st.subheader("1분 예측 경로 (10초 칸 6개: 10·20·…·60초 뒤의 10초 평균)")
    b10 = pd.read_csv(bp, encoding="utf-8-sig")
    names = {"actual": "실제", "persistence": "직전값", "ridge_direct": "ridge", "lightgbm_direct": "LightGBM",
             "lightgbm_recursive": "LightGBM(10초×6 반복)", "lstm_direct_s0": "LSTM", "lstm_recursive_s0": "LSTM(10초×6 반복)", "avg_direct_s0": "LightGBM+LSTM"}
    pick = st.multiselect("경로 모델", list(names)[1:], default=["persistence", "lightgbm_direct", "lstm_direct_s0", "avg_direct_s0"], format_func=names.get)
    t0 = int(b10.last_observed_sec.min())
    if sec < t0:
        st.info(f"사이드바 '영상 시각'을 {t0}초 이후로 옮기면 그 시각에서 낸 1분 예측 경로가 보입니다(평가 구간만 저장).")
    else:
        cols = st.columns(len(zones))
        for col, z in zip(cols, zones):
            g = b10[(b10.zone_id == z["name"]) & (b10.last_observed_sec == sec)].set_index("horizon_s")
            now = obs[(obs.zone == z["name"]) & (obs.sec.between(sec - 9, sec))]["count"].mean()
            path = pd.concat([pd.DataFrame({k: [now] for k in ["actual"] + pick}, index=[0]), g[["actual"] + pick]]).rename(columns=names)
            path.index = sec + path.index
            col.caption(f"{z['name']} — {sec}초에서 낸 예측 (0 = 지금 10초 평균)")
            col.line_chart(path, height=220, x_label="영상 시각(초)", y_label="명")
    fx10 = Path(run["zones_dir"]) / "bins10_compare.csv"
    if fx10.exists():
        t = pd.read_csv(fx10, encoding="utf-8-sig")
        t["MAE_sd"] = t["MAE_sd"].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        st.markdown("**10초 칸 방식 비교 (뒤 40%, 같은 시점)** — jump = 1초마다 예측값이 바뀌는 평균 크기(작을수록 매끈, 실제값 ≈ 0.12명). "
                    "`*_direct` = 칸마다 바로 예측, `*_recursive` = 10초 예측을 6번 반복, `1s_*` = 기존 1초 방식(10·30·60초만)")
        st.dataframe(t.rename(columns={"horizon_s": "예측 시간(초)", "model": "모델", "MAE_sd": "MAE 표준편차", "skill": "직전값 대비 개선율",
                                       "jump": "흔들림(jump)", "n": "비교 시점 수", "seeds": "시드 수"}), hide_index=True, width="stretch", height=300)
        st.dataframe(pd.read_csv(fx10.with_name("bins10_bootstrap.csv"), encoding="utf-8-sig").rename(columns={"horizon_s": "예측 시간(초)"}),
                     hide_index=True, width="stretch", height=250)

st.subheader("예측 성능 (걷기식 백테스트, MAE는 낮을수록 좋음)")
if len(fc):
    from forecast.serving.video_pipeline import skill_table
    sk = pd.DataFrame(skill_table(fc)).T.rename_axis(["예측 시간(초)", "모델"]).rename(columns={"skill_vs_persistence": "직전값 대비 개선율", "n": "비교 시점 수"})
    st.dataframe(sk[sk["비교 시점 수"] > 0], width="stretch")
    st.caption(f"개선율 > 0 이면 직전값(persistence)보다 낫다. 영상 {secs + 1}초 전체 구간 기준(모델마다 예측을 시작하는 시점이 달라 비교 시점 수가 다르다).")
    cut = run.get("test_from_sec")
    if cut is not None:
        st.subheader(f"공정 비교: {cut}초 이후(미리 학습 모델이 보지 않은 구간)만, 모든 모델이 같은 시점")
        fair = fc[fc.last_observed_sec >= cut]
        fair = fair[fair.groupby(["horizon_s", "zone_id", "last_observed_sec"]).status.transform(lambda s: (s == "ok").all())]
        fk = pd.DataFrame(skill_table(fair)).T.rename_axis(["예측 시간(초)", "모델"]).rename(columns={"skill_vs_persistence": "직전값 대비 개선율", "n": "비교 시점 수"})
        st.dataframe(fk, width="stretch")
        st.caption(f"lightgbm_pre·lstm_pre는 0~{cut - 1}초만으로 학습(같은 카메라·같은 날). lightgbm·ridge_ar는 매 시점 그때까지 본 영상으로 다시 학습(걷기식). "
                   "lstm_pre = 아래 같은 조건 비교의 lstm 시드 0과 같은 모델.")
    zd = Path(run["zones_dir"])
    if (zd / "fixed_compare.csv").exists():  # 실험 스크립트(fixed_test.py) 결과가 있을 때만
        st.subheader("같은 조건 비교: 모두 앞 60%로만 학습 → 뒤 40% 평가 (LSTM·결합은 시드 3개 평균±표준편차)")
        fx = pd.read_csv(zd / "fixed_compare.csv", encoding="utf-8-sig")
        for col in ("MAE_sd", "skill_sd", "lgbm_weight"):  # 시드 1개 모델은 표준편차가 없다 -> None 대신 빈칸
            fx[col] = fx[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        st.dataframe(fx.rename(columns={"horizon_s": "예측 시간(초)", "model": "모델", "MAE_sd": "MAE 표준편차", "skill": "직전값 대비 개선율",
                                        "skill_sd": "개선율 표준편차", "n": "비교 시점 수", "seeds": "시드 수", "lgbm_weight": "결합 시 LightGBM 몫"}),
                     hide_index=True, width="stretch")
        st.caption("avg = LightGBM·LSTM 단순 평균, wavg = 학습 구간 마지막 20%에서 고른 가중 평균. walk_* = 위 걷기식 결과(참고, 계속 다시 학습).")
        st.markdown("**차이가 우연인지 (60초 블록 부트스트랩 2,000회, MAE 차이 95% 구간 — 0을 포함하면 차이 불확실)**")
        st.dataframe(pd.read_csv(zd / "fixed_bootstrap.csv", encoding="utf-8-sig").rename(columns={"horizon_s": "예측 시간(초)"}), hide_index=True, width="stretch")
    miss = fc[fc.status != "ok"].groupby(["model", "status"]).size()
    if len(miss):
        st.caption("산출 안 된 모델: " + ", ".join(f"{m} {s} {n}건" for (m, s), n in miss.items()) + " (LightGBM은 학습 행 200개 이상 ≈ 4분 이상 영상 필요)")
st.subheader("이상 흐름 단계 타임라인 (0 낮음 · 1 주의 · 2 높음)")
if len(tl):
    st.bar_chart(tl.pivot_table(index="sec", columns="zone_id", values="level_num"), stack=False, height=180, x_label="영상 시각(초)")
    st.caption("기준: 높음 = 병목 70 이상 / 규칙 경보와 Isolation Forest 동시 / 주의 20초 연속. 주의 = 병목 40~70 / 규칙 경보 / IF 점수 0.8 이상. 3초 유지해야 올리고 5초 풀려야 내린다. 기준값은 제안값.")
lv = an[an.method == "level"] if len(an) and "method" in an else an
n_rule = int((an.method == "rule").sum()) if len(an) else 0
n_if = int((an.method == "iforest").sum()) if len(an) else 0
st.subheader(f"단계 이벤트 {len(lv)}건 (세부: 규칙 {n_rule}건, Isolation Forest {n_if}건)")
if len(lv):
    show = lv[["zone_id", "level", "event_sec", "end_sec", "duration_s", "peak_sec", "reason"]].rename(columns={"zone_id": "구역", "level": "단계", "event_sec": "시작(초)", "end_sec": "끝(초)", "duration_s": "지속(초)", "peak_sec": "최고 시각", "reason": "근거"})
    st.dataframe(show, hide_index=True, width="stretch")
    pick = st.selectbox("이벤트 상세 (평소 대비 편차 z)", range(len(lv)), format_func=lambda i: f"{lv.zone_id.iloc[i]} {lv.event_sec.iloc[i]}~{lv.end_sec.iloc[i]}초 {lv.level.iloc[i]}")
    names = {"count_delta_5s": "인원 변화(5초)", "motion_drop": "이동량 감소", "reverse_ratio": "역방향", "dwell": "거의 서 있음", "if_score": "IF 점수", "bottleneck_score": "병목"}
    zf = pd.Series(json.loads(lv.features.iloc[pick])).rename(names)
    st.bar_chart(zf, height=200, y_label="평소 대비 편차(z)")
    st.caption(f"최고 시각 {lv.peak_sec.iloc[pick]}초. 영상으로 보려면 위 슬라이더를 {max(0, int(lv.event_sec.iloc[pick]) - 5)}초(시작 5초 전)로 옮기세요.")
if len(bn):
    st.subheader("병목 점수 (팀 Vision 파이프라인, 10초 윈도우, 참고)")
    st.bar_chart(bn.pivot_table(index="window_start", columns="zone", values="score"), stack=False, height=200)
