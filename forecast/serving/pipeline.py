"""서울 생활인구 CSV -> 전처리 -> 1시간 뒤 예측(Seasonal Naive, LightGBM, LSTM) -> 결과 CSV. 재학습 없이 artifacts의 모델을 쓴다.

Python:
    from forecast.serving import run_forecast
    res = run_forecast("250_LOCAL_RESD_ADMDONG_202610")            # 데이터 마지막 시각 + 1시간 한 시점
    res = run_forecast(["...202609", "...202610"], start="2026-10-01 00:00", end="2026-10-31 23:00")  # 구간(실측 비교용)
    res["path"], res["predictions"], res["info"]
CLI (저장소 루트에서): python -m forecast.serving.pipeline 입력경로 [입력경로 ...] [--out 결과.csv] [--start ...] [--end ...]

입력: 서울 생활인구(OA-23016) 행정동 원본 CSV(파일·폴더·ZIP·목록) 또는 population_history 형식 CSV(observed_at, area_id, estimate).
최소 168시간(7일)의 연속 이력이 있어야 LightGBM·LSTM이 예측한다. 결과 컬럼은 README.md의 스키마 표 참고."""
import argparse, io, json
from functools import lru_cache
from pathlib import Path
import numpy as np, pandas as pd

from ..data_pipeline.living_pop_io import iter_sources, parse, read_csv_bytes, to_wide
from .predict import A, L, Predictor

TZ = "Asia/Seoul"
MODELS = ("seasonal_naive", "lightgbm", "lstm")
VERSIONS = {k: v["version"] for k, v in json.loads((A / "manifest.json").read_text(encoding="utf-8"))["models"].items()}
COLUMNS = ["area_id", "model", "model_version", "last_observed_at", "target_at", "value", "actual", "status"]


@lru_cache(maxsize=4)
def get_predictor(device="cpu", lgbm_path=None):
    """모델은 한 번만 읽는다(FastAPI에서 반복 호출해도 재로드 없음)."""
    return Predictor(lgbm_path=lgbm_path, device=device)


def _read_population_history(b):
    d = pd.read_csv(io.BytesIO(b), dtype={"area_id": str}, encoding="utf-8-sig")
    t = pd.to_datetime(d.observed_at)
    t = t.dt.tz_convert(TZ).dt.tz_localize(None) if t.dt.tz is not None else t  # 시간대가 있으면 한국 시각으로 맞춘다
    return pd.DataFrame({"datetime": t, "adm_cd": d.area_id.str.strip(), "total_pop": pd.to_numeric(d.estimate, errors="coerce")})


def load_history(source):
    """입력 -> (wide[1시간 간격 × 행정동], info). 원본 CSV는 학습 데이터와 같은 규칙(living_pop_io)으로 정제한다."""
    raw, ph, names = [], [], []
    for name, b in iter_sources(source):
        names.append(name)
        head = b[:300].decode("utf-8-sig", errors="ignore")
        if {"sec", "zone", "count"} <= set(head.splitlines()[0].replace(" ", "").split(",")):
            raise ValueError(f"{name}: 영상 구역 CSV(sec, zone, count…)는 이 예측의 입력이 아닙니다. 서울 생활인구 CSV를 넣으세요. "
                             "영상 CSV는 이상 흐름 탐지(forecast/anomaly_video.py)에 씁니다.")
        (ph.append(_read_population_history(b)) if "observed_at" in head and "area_id" in head else raw.append(read_csv_bytes(b)))
    if not names:
        raise FileNotFoundError(f"CSV가 없습니다: {source}")
    parts = [parse(raw)[0]] if raw else []
    long = pd.concat(parts + ph, ignore_index=True).drop_duplicates(["datetime", "adm_cd"], keep="last")
    wide, missing, imputed = to_wide(long)
    wide.columns = wide.columns.astype(str)
    info = {"input_files": len(names), "input_rows": int(len(long)), "observed_from": wide.index.min(), "observed_to": wide.index.max(),
            "areas_in_input": int(wide.shape[1]), "imputed_cells": int(imputed.to_numpy().sum()), "still_missing_cells": int(wide.isna().to_numpy().sum())}
    return wide, info


def _naive(wide, T):
    """Seasonal Naive: 지난주 같은 시각 실측. 없으면 NaN과 사유."""
    src = T - pd.Timedelta(hours=L)
    if src < wide.index.min():
        return np.full(wide.shape[1], np.nan), np.full(wide.shape[1], "insufficient_history", dtype=object)
    v = wide.loc[src].to_numpy(float)
    return v, np.where(np.isnan(v), "gap_in_history", "ok")


def run_forecast(source, out_csv=None, start=None, end=None, models=MODELS, device="cpu", lgbm_path=None):
    """source: CSV 파일/폴더/ZIP 또는 목록. start·end를 주지 않으면 데이터 마지막 시각 + 1시간 한 시점을 예측한다.
    start·end를 주면 그 사이의 매시각을 '그 1시간 전까지의 실측'으로 예측한다(1단계 롤링, actual 채워짐).
    반환: {"path": 결과 CSV 경로, "predictions": DataFrame(COLUMNS), "info": 요약}"""
    bad = set(models) - set(MODELS)
    if bad:
        raise ValueError(f"모르는 모델: {sorted(bad)} (가능: {MODELS})")
    wide, info = load_history(source)
    last = wide.index.max()
    lo, hi = wide.index.min() + pd.Timedelta(hours=1), last + pd.Timedelta(hours=1)
    if start is None and end is None:
        targets = pd.DatetimeIndex([hi])
    else:
        s, e = pd.Timestamp(start) if start else lo, pd.Timestamp(end) if end else hi
        if s > e or s < lo or e > hi:
            raise ValueError(f"예측 대상 구간은 {lo} ~ {hi} 안이어야 합니다(1시간 앞까지만 예측). 받은 값: {s} ~ {e}")
        targets = pd.date_range(s, e, freq="h")
    pred = get_predictor(device, lgbm_path) if {"lightgbm", "lstm"} & set(models) else None
    rows = []
    for T in targets:
        hist = wide.loc[:T - pd.Timedelta(hours=1)].iloc[-(L + 1):]
        actual = wide.loc[T] if T in wide.index else pd.Series(np.nan, index=wide.columns)
        base = {"last_observed_at": T - pd.Timedelta(hours=1), "target_at": T}
        if "seasonal_naive" in models:
            v, st = _naive(wide, T)
            rows.append(pd.DataFrame({"area_id": wide.columns, "model": "seasonal_naive", "value": v, "actual": actual.to_numpy(float), "status": st, **base}))
        if pred is not None:
            out = pred.predict(hist, T)
            act = actual.reindex(out.adm_cd).to_numpy(float)
            for m in ("lightgbm", "lstm"):
                if m in models:
                    rows.append(pd.DataFrame({"area_id": out.adm_cd, "model": m, "value": out[m].to_numpy(), "actual": act, "status": out.status.to_numpy(), **base}))
    df = pd.concat(rows, ignore_index=True)
    df["model_version"] = df.model.map(VERSIONS)
    for c in ("last_observed_at", "target_at"):
        df[c] = pd.to_datetime(df[c]).dt.tz_localize(TZ)
    df["value"], df["actual"] = df.value.astype("float64").round(1), df.actual.astype("float64").round(1)
    df = df[COLUMNS].astype({"area_id": "string", "model": "string", "model_version": "string", "status": "string"})
    df = df.sort_values(["target_at", "model", "area_id"]).reset_index(drop=True)
    if out_csv is None:
        first = Path(source[0] if isinstance(source, (list, tuple)) else source)
        out_csv = (first if first.is_dir() else first.parent).parent / f"forecast_{targets[0]:%Y%m%d%H}_{targets[-1]:%Y%m%d%H}.csv"
    out = df.copy()
    for c in ("last_observed_at", "target_at"):
        out[c] = out[c].map(lambda x: x.isoformat())  # 2026-10-14T08:00:00+09:00
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    info.update({"targets_from": targets[0], "targets_to": targets[-1], "n_targets": len(targets), "rows": len(df),
                 "status_counts": df.groupby(["model", "status"]).size().unstack(fill_value=0).to_dict("index"),
                 "model_versions": {m: VERSIONS[m] for m in models}})
    return {"path": str(out_csv), "predictions": df, "info": info}


def main():
    ap = argparse.ArgumentParser(description="서울 생활인구 CSV로 1시간 뒤 인구를 예측해 결과 CSV를 만든다.")
    ap.add_argument("source", nargs="+", help="생활인구 CSV 파일·폴더·ZIP (여러 개 가능)")
    ap.add_argument("--out", help="결과 CSV 경로 (생략 시 입력 옆에 생성)")
    ap.add_argument("--start", help="구간 예측 시작 시각 (예: 2026-10-01 00:00)")
    ap.add_argument("--end", help="구간 예측 끝 시각")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    res = run_forecast(a.source if len(a.source) > 1 else a.source[0], a.out, a.start, a.end, tuple(a.models), a.device)
    i = res["info"]
    print(f"결과: {res['path']} ({i['rows']:,}행) | 실측 {i['observed_from']} ~ {i['observed_to']} | 예측 대상 {i['targets_from']} ~ {i['targets_to']}")
    print(pd.DataFrame(i["status_counts"]).T.to_string())


if __name__ == "__main__":
    main()
