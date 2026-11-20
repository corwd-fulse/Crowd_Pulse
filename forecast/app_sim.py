"""JuPedSim 시나리오 5/10/15분 예측 뷰어: 장면(사람 위치·속도) + 실제 vs 모델 예측 + 결과 표.
실행(data-area 루트, 시스템 python): streamlit run forecast/app_sim.py --server.port 8508
필요: data/simcrowd/(scenarios.csv, parts/, traj/ — sim_dataset --record로 생성) + forecast/bottleneck/runs/<실행>/(sim_models 결과)
시뮬레이션 데이터입니다. 실제 성능이 아니라 모델 비교·공간 특성 효과 확인용입니다."""
import json, re, sys, time
from pathlib import Path

# 앱 폴더(forecast/)의 bottleneck 패키지가 pandas의 선택 의존성 'bottleneck'으로 잘못 불러와지지 않게
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
import altair as alt, matplotlib.pyplot as plt, numpy as np, pandas as pd, streamlit as st

D = Path(__file__).resolve().parents[1] / "data" / "simcrowd"
RUNS = Path(__file__).resolve().parent / "bottleneck" / "runs"
SHAPE = {"straight": "직선 통로", "bottleneck": "병목(좁은 문)", "tjunction": "T자 합류"}
plt.rcParams["font.family"] = "Malgun Gothic"

st.set_page_config(page_title="시뮬레이션 혼잡 예측", layout="wide")
st.title("JuPedSim 시나리오 · 5/10/15분 뒤 구역 인원 예측")
st.caption("시뮬레이션 데이터라서 실제 성능이 아닙니다. 모델끼리 비교하고 공간 특성의 효과를 보는 용도입니다. 테스트 시나리오는 학습에 쓰지 않은 폭·형태 조합입니다.")


def mtime(p):
    return p.stat().st_mtime if p.exists() else 0


@st.cache_data
def load(run, _k):
    return pd.read_csv(D / "scenarios.csv"), pd.read_parquet(run / "test_predictions.parquet")


@st.cache_data
def seconds(sid, _k):
    return pd.read_parquet(D / "parts" / f"{sid:03d}.parquet")


@st.cache_data
def traj(sid, _k):
    t = pd.read_parquet(D / "traj" / f"{sid:03d}.parquet")
    g = json.loads((D / "traj" / f"{sid:03d}.json").read_text())
    return t, g


def ring(wkt):
    """POLYGON WKT 바깥 테두리 좌표 (shapely 없이)."""
    first = wkt.split("),")[0]
    return np.array([[float(a), float(b)] for a, b in re.findall(r"(-?[\d.e+-]+) (-?[\d.e+-]+)", first)])


def speeds(tr, t):
    now, before = tr[tr.t == t - 1], tr[tr.t == t - 2].set_index("id")
    b = before.reindex(now.id)
    return now, np.nan_to_num(np.hypot(now.x.to_numpy() - b.x.to_numpy(), now.y.to_numpy() - b.y.to_numpy()), nan=1.0)


def scene(sid, k, ts):
    """시각 t의 장면(통로·구역·출구 + 사람 점, 색 = 속도). 반환 (fig, scatter, title)."""
    tr, g = traj(sid, k)
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, kw in (("geometry", dict(color="0.3", lw=1.5)), ("zone", dict(color="tab:blue", lw=1, ls="--")), ("exit", dict(color="tab:green", lw=2))):
        xy = ring(g[key]); ax.plot(xy[:, 0], xy[:, 1], **kw)
    now, sp = speeds(tr, ts[0])
    sca = ax.scatter(now.x, now.y, c=sp, cmap="RdYlGn", vmin=0, vmax=1.5, s=6)
    fig.colorbar(sca, ax=ax, label="속도 (m/s)")
    ax.set_aspect("equal")
    ttl = ax.set_title(f"{ts[0] // 60}분 {ts[0] % 60}초 · 점선 = 구역, 초록 = 출구", fontsize=9)
    return fig, sca, ttl, tr


runs = sorted((r for r in RUNS.glob("*") if (r / "test_predictions.parquet").exists()), reverse=True)
if not runs:
    st.warning("아직 결과가 없습니다. sim_models --out <이름>을 먼저 실행하세요.")
    st.stop()
R = RUNS / st.sidebar.selectbox("실행 결과 (runs/)", [r.name for r in runs])
sc, P = load(R, mtime(R / "test_predictions.parquet"))
models = [c for c in P.columns if c not in ("sid", "shape", "t", "part", "horizon", "n0", "actual")]

tab1, tab2 = st.tabs(["시나리오 보기", "결과 표"])
with tab2:
    if (R / "results.md").exists():
        st.markdown((R / "results.md").read_text(encoding="utf-8"))

with tab1:
    test = sc[sc.sid.isin(P.sid.unique())].copy()
    test["설명"] = test.apply(lambda r: f"{r.sid}번 · {SHAPE[r['shape']]} · 폭 {r.width:.1f}m · 출구 {r.exit_width:.1f}m"
                              + (" · 위치 기록 있음" if (D / "traj" / f"{r.sid:03d}.parquet").exists() else ""), axis=1)
    c1, c2, c3 = st.columns([3, 1, 3])
    pick = c1.selectbox("테스트 시나리오", test["설명"])
    sid = int(pick.split("번")[0])
    h = c2.radio("몇 분 뒤", [5, 10, 15], horizontal=True) * 60
    show = c3.multiselect("모델", models, default=[m for m in models if m in ("직전값", "LightGBM", "앙상블 가중")])
    s = seconds(sid, mtime(D / "parts" / f"{sid:03d}.parquet"))
    p = P[(P.sid == sid) & (P.horizon == h)]

    has_traj = (D / "traj" / f"{sid:03d}.parquet").exists()
    tmin, tmax, key = int(p.t.min()), int(p.t.max()), f"t_{sid}"
    if "_next_t" in st.session_state:  # 재생 중: 슬라이더가 그려지기 전에 다음 시각으로
        st.session_state[key] = int(st.session_state.pop("_next_t"))
    st.session_state.setdefault(key, tmin)
    pc = st.columns([1, 1, 4])
    playing = st.session_state.get("playing", False)
    if pc[0].button("⏸ 멈춤" if playing else "▶ 재생", key="play_btn", width="stretch"):  # 고정 키: 글자가 바뀌어도 같은 버튼(흐린 잔상 방지)
        st.session_state.playing = playing = not playing
    jump = pc[1].selectbox("재생 속도", [30, 60, 120], format_func=lambda x: f"한 번에 {x}초", label_visibility="collapsed")
    t_now = pc[2].slider("지금 시각(초)", tmin, tmax, step=30, key=key, label_visibility="collapsed")
    st.caption(f"지금 {t_now // 60}분 {t_now % 60}초 · ▶ 재생을 누르면 시각이 저절로 넘어가며 그 시점까지 본 것만으로 낸 예측이 갱신됩니다.")
    left, right = st.columns([1, 1])

    with left:
        st.subheader("장면")
        if has_traj:
            fig = scene(sid, mtime(D / "traj" / f"{sid:03d}.parquet"), [t_now])[0]
            st.pyplot(fig); plt.close(fig)
        else:
            st.info("이 시나리오는 위치 기록이 없습니다: python -m forecast.bottleneck.sim_dataset --record " + str(sid))
        row = s[s.t == t_now - 1].iloc[0]
        r = sc.set_index("sid").loc[sid]
        m = st.columns(4)
        m[0].metric("구역 인원", f"{int(row.n)}명"); m[1].metric("밀도", f"{row.density:.2f}명/㎡")
        m[2].metric("평균 속도", f"{row.speed:.2f}m/s"); m[3].metric("입구 밖 대기", f"{int(row.queue_outside)}명")
        pn = P[(P.sid == sid) & (P.t == t_now)]
        if len(pn):
            st.markdown("**지금 시각에서 낸 예측 (명)**")
            tbl = pn.set_index("horizon")[["actual"] + show].rename(columns={"actual": "실제"})
            tbl.index = [f"{x // 60}분 뒤" for x in tbl.index]
            st.dataframe(tbl.round(0), width="stretch")

    with right:
        st.subheader(f"실제 vs {h // 60}분 뒤 예측")
        seen = s[s.t < t_now]  # 지금까지 들어온 실제 값만
        actual = seen[["t", "n"]].assign(t=seen.t + 1).rename(columns={"n": "인원"}).assign(계열="실제").iloc[::10]
        pr = p[p.t <= t_now].assign(t=lambda d: d.t + h).melt("t", show, "계열", "인원")  # 지금까지 낸 예측(맞히려는 시각에 그림)
        lines = pd.concat([actual, pr])
        ymax = float(max(s.n.max(), p[show].max().max() if show else 0)) * 1.05 + 1
        ch = alt.Chart(lines).mark_line().encode(x=alt.X("t:Q", title="시각(초)", scale=alt.Scale(domain=[0, 3600 + h])),
                                                 y=alt.Y("인원:Q", title="구역 인원(명)", scale=alt.Scale(domain=[0, ymax])),
                                                 color=alt.Color("계열:N", legend=alt.Legend(orient="bottom")))
        rule = alt.Chart(pd.DataFrame({"t": [t_now]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(x="t:Q")
        st.altair_chart(ch + rule, width="stretch")
        st.caption(f"예측선은 맞히려는 시각(예측한 시각 + {h // 60}분)에 그렸습니다. 회색 점선 = 지금, 점선 오른쪽 = 아직 안 온 미래에 대한 예측.")
        err = pd.DataFrame({m_: [(p[m_] - p.actual).abs().mean()] for m_ in show}, index=["이 시나리오 MAE(명)"]).round(2)
        st.dataframe(err, width="stretch")
        st.caption(f"시나리오 정보: {SHAPE[r['shape']]}, 통로 폭 {r.width:.1f}m, 출구 폭 {r.exit_width:.1f}m, 면적 {r.area:.0f}㎡"
                   + (f", 가로 통로 폭 {r.branch_width:.1f}m" if r['shape'] == 'tjunction' else ""))

    if playing:  # 다음 시각으로 넘어가 화면 전체를 다시 그림
        if t_now + jump > tmax:  # 끝: 멈추고 버튼 글자를 '재생'으로 다시 그림
            st.session_state.playing = False
            st.rerun()
        else:
            time.sleep(0.6)
            st.session_state["_next_t"] = t_now + jump
            st.rerun()
