"""서울 생활인구(OA-23016) 전처리. 사용: python preprocess_living_pop.py [--raw DIR] [--out DIR] [--dongs 코드 ...]
raw_dir 아래 월별 폴더(250_LOCAL_RESD_ADMDONG_YYYYMM)나 ZIP을 읽는다. ' (1)' 중복본은 무시.
읽기·정제 규칙은 living_pop_io.py에 있다(추론 파이프라인과 공용)."""
import argparse, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from living_pop_io import parse, read_csv_bytes, sources, to_wide

ap = argparse.ArgumentParser()
ap.add_argument("--raw", type=Path, default=Path(__file__).resolve().parents[2])
ap.add_argument("--out", type=Path)
ap.add_argument("--dongs", nargs="+", help="대상 행정동코드만 남김 (생략 시 서울 전체)")
args = ap.parse_args()
RAW = args.raw
OUT = args.out or RAW / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

frames = []
for name, b in sources(RAW):
    d = read_csv_bytes(b)
    frames.append(d)
    print(f"{name}: {len(d):,} rows")
df, dup_n, n0 = parse(frames)
print(f"중복 제거: {n0:,} -> {len(df):,}")

if args.dongs:
    df = df[df["adm_cd"].isin(args.dongs)]
    assert not df.empty, f"해당 행정동 없음: {args.dongs}"
wide, missing, imputed = to_wide(df)

long = wide.stack().rename("total_pop").reset_index()
long["is_imputed"] = imputed.stack().reindex(pd.MultiIndex.from_frame(long[["datetime", "adm_cd"]])).to_numpy()
long = long[["datetime", "adm_cd", "total_pop", "is_imputed"]]

q = pd.DataFrame({
    "n_expected": len(wide),
    "n_missing_raw": missing.sum(),
    "n_imputed": imputed.sum(),
    "n_still_missing": wide.isna().sum(),
    "n_zero": (wide == 0).sum(),
    "n_duplicates": dup_n.reindex(wide.columns).fillna(0).astype(int),
    "max_pop": wide.max(),
    "min_pop": wide.min(),
    "mean_pop": wide.mean().round(1),
})
q.index.name = "adm_cd"
long.to_parquet(OUT / "living_pop_long.parquet", index=False)
wide.to_parquet(OUT / "living_pop_wide.parquet")
q.to_csv(OUT / "quality_report.csv", encoding="utf-8-sig")
print(f"기간 {wide.index.min()} ~ {wide.index.max()}, 행정동 {wide.shape[1]}, 시점 {len(wide):,}")
print(q.describe().loc[["min", "max", "mean"]].round(1).to_string())
