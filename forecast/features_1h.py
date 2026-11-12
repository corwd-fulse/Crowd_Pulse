"""1시간 뒤 예측용 feature 생성(train_1h.py, verify_saved.py, analyze_1h.py 공용).
모든 feature는 T-1h 이전 값만 사용. load_features(w)에 다른 데이터를 넣어 미래값 교란 테스트도 가능."""
from pathlib import Path
import pandas as pd, holidays

ROOT = Path(__file__).resolve().parents[1]
VAL_START, TEST_START = "2026-09-01", "2026-10-01"
LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 144, 168)
HOL = set(pd.to_datetime(list(holidays.KR(years=[2025, 2026]))).date) | {pd.Timestamp("2026-09-28").date()}  # 추석 대체공휴일 수동 추가


def load_wide():
    return pd.read_parquet(ROOT / "data/processed/living_pop_wide.parquet")


def load_features(w=None, scale_end=VAL_START):
    """(scale_end: 동별 규모 보정에 쓰는 구간의 끝, 롤링 검증에서는 fold마다 바꿈) 반환: d(전체 행), cols(feature 이름), scale(동별 학습구간 평균), (tr, va, te)"""
    w = load_wide() if w is None else w
    scale = w.loc[:scale_end].iloc[:-1].mean()  # 동별 규모 보정은 학습구간 평균만 사용
    r = w / scale

    feat = {f"lag_{k}h": r.shift(k) for k in LAGS}
    feat["diff_1h"] = r.shift(1) - r.shift(2)
    feat["diff_3h"] = r.shift(1) - r.shift(4)
    for n in (24, 168):
        feat[f"mean_{n}h"] = r.rolling(n).mean().shift(1)
        feat[f"std_{n}h"] = r.rolling(n).std().shift(1)
    feat["surge_vs_week"] = r.shift(1) / r.shift(169)  # 직전 시간이 지난주 같은 시각 대비 얼마나 높았나(행사 신호)

    d = pd.DataFrame({k: v.stack(future_stack=True) for k, v in feat.items()})
    d["y"] = r.stack(future_stack=True)
    d = d.reset_index()
    hs = pd.Series(pd.Series(w.index.date, w.index).isin(HOL).astype(int).to_numpy(), w.index)
    d["is_hol"] = d["datetime"].map(hs)
    d["is_hol_next"] = (d["datetime"] + pd.Timedelta(days=1)).map(hs).fillna(0)
    d["is_hol_prev"] = (d["datetime"] - pd.Timedelta(days=1)).map(hs)
    d["hour"] = d["datetime"].dt.hour
    d["dow"] = d["datetime"].dt.day_of_week
    d["dong"] = d["adm_cd"].astype("category")
    d = d.dropna(subset=["y", "lag_168h", "is_hol_prev"]).reset_index(drop=True)  # surge_vs_week는 NaN 허용

    cols = [c for c in d.columns if c not in ("datetime", "adm_cd", "y")]
    tr, va, te = (d[d.datetime < VAL_START], d[(d.datetime >= VAL_START) & (d.datetime < TEST_START)], d[d.datetime >= TEST_START])
    return d, cols, scale, (tr, va, te)
