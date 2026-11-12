"""run_forecast 검증. 저장소 루트에서: python -m forecast.serving.test_pipeline
필요: 원본 폴더 250_LOCAL_RESD_ADMDONG_202609, 202610 과 학습 때 저장한 forecast/pred_test_1h.parquet(일치 비교용)."""
import tempfile
from pathlib import Path
import numpy as np, pandas as pd

from .pipeline import COLUMNS, load_history, run_forecast

ROOT = Path(__file__).resolve().parents[2]
SEP, OCT = ROOT / "250_LOCAL_RESD_ADMDONG_202609", ROOT / "250_LOCAL_RESD_ADMDONG_202610"
TMP = Path(tempfile.mkdtemp(prefix="fc_test_"))
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(("통과 " if cond else "실패 ") + name + (f" — {detail}" if detail else ""))


def as_ph_csv(wide, path):
    """wide -> population_history 형식 CSV (DB에서 뽑은 경우를 흉내)"""
    long = wide.rename_axis("observed_at").reset_index().melt("observed_at", var_name="area_id", value_name="estimate").dropna()
    long["observed_at"] = pd.to_datetime(long.observed_at).dt.tz_localize("Asia/Seoul").map(lambda x: x.isoformat())
    long.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# 1) 원본 CSV -> 학습 때 저장한 10월 예측과 일치 (공휴일 10/3 하루 + 10/9 아침)
ref = pd.read_parquet(ROOT / "forecast/pred_test_1h.parquet"); ref["adm_cd"] = ref.adm_cd.astype(str)
ref["datetime"] = pd.to_datetime(ref.datetime).dt.tz_localize("Asia/Seoul")
parts = [run_forecast([SEP, OCT], TMP / f"range{i}.csv", s, e)["predictions"] for i, (s, e) in enumerate((("2026-10-03 00:00", "2026-10-03 23:00"), ("2026-10-09 06:00", "2026-10-09 11:00")))]
p = pd.concat(parts)
w = p.pivot_table(index=["target_at", "area_id"], columns="model", values="value").join(p[p.model == "lstm"].set_index(["target_at", "area_id"]).actual)
r = ref.set_index(["datetime", "adm_cd"]).reindex(w.index)
d = {m: float((w[m] - r[c]).abs().max()) for m, c in (("seasonal_naive", "naive"), ("lightgbm", "lightgbm"))}
rel = float(((w.lstm - r.lstm).abs() / r.actual).max())
mae_new, mae_ref = float((w.lstm - r.actual).abs().mean()), float((r.lstm - r.actual).abs().mean())
check("실측(actual)이 저장된 값과 같음", float((w.actual - r.actual).abs().max()) <= 0.1 + 1e-6, f"최대 차이 {float((w.actual - r.actual).abs().max()):.3f}명")
check("Seasonal Naive 일치", d["seasonal_naive"] <= 0.1 + 1e-6, f"최대 차이 {d['seasonal_naive']:.3f}명")
check("LightGBM 일치(반올림 범위)", d["lightgbm"] <= 0.1 + 1e-6, f"최대 차이 {d['lightgbm']:.3f}명")
check("LSTM 일치(저장값 fp16 잡음 범위)", rel < 3e-3 and mae_new <= mae_ref + 0.5, f"상대 최대 {rel * 100:.3f}%, 실측 대비 MAE 파이프라인 {mae_new:.2f} / 저장값 {mae_ref:.2f}")
check("비교 건수", len(w) == 30 * 427, f"{len(w):,}행(30시각 × 427동)")

# 2) 스키마
check("컬럼 순서", list(p.columns) == COLUMNS, str(list(p.columns)))
check("시각 타입·시간대", str(p.target_at.dtype) == "datetime64[ns, Asia/Seoul]" or "Asia/Seoul" in str(p.target_at.dtype), str(p.target_at.dtype))
check("target_at = last_observed_at + 1시간", ((p.target_at - p.last_observed_at) == pd.Timedelta(hours=1)).all())
check("(area_id, model, target_at) 중복 없음", not p.duplicated(["area_id", "model", "target_at"]).any())
c = pd.read_csv(TMP / "range0.csv", dtype={"area_id": str})
check("CSV 시각이 ISO 8601(+09:00)", c.target_at.str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:00:00\+09:00$").all(), c.target_at.iloc[0])

# 3) 기본 모드: 10월 원본만 -> 11/01 00시 한 시점
res = run_forecast(OCT, TMP / "default.csv")
q = res["predictions"]
check("기본 모드 1,281행(427동 × 3모델), 모두 ok", len(q) == 1281 and (q.status == "ok").all(), f"{len(q)}행, 대상 {q.target_at.iloc[0]}")
check("실측 마지막 시각 포함", (q.last_observed_at == pd.Timestamp("2026-10-31 23:00", tz="Asia/Seoul")).all())

# 4) 예외: 이력 부족(10/01~10/05만), 모르는 동, 3시간 결측, 입력 형식 동등성
short = run_forecast(sorted(OCT.glob("*.csv"))[:5], TMP / "short.csv")["predictions"]
check("이력 168시간 미만 -> insufficient_history, 값 비움", (short.status == "insufficient_history").all() and short.value.isna().all(), short.status.value_counts().to_dict())
wide, _ = load_history(OCT)
x = wide.copy(); x["99999999"] = x.iloc[:, 0]; dong = x.columns[5]; x.loc[x.index[-6:-3], dong] = np.nan  # 3시간 결측(보간 한도 2시간 초과)
z = run_forecast(as_ph_csv(x, TMP / "ph_edge.csv"), TMP / "edge.csv")["predictions"]
st = z.set_index(["model", "area_id"]).status
check("427개 밖 동 -> 모델은 unknown_area, Seasonal Naive는 계산", st[("lstm", "99999999")] == "unknown_area" and st[("lightgbm", "99999999")] == "unknown_area" and st[("seasonal_naive", "99999999")] == "ok")
check("최근 168시간 안 3시간 결측 -> gap_in_history", st[("lstm", dong)] == "gap_in_history" and st[("lightgbm", dong)] == "gap_in_history", f"동 {dong}")
same = run_forecast(as_ph_csv(wide, TMP / "ph_full.csv"), TMP / "ph.csv")["predictions"]
check("원본 CSV와 population_history CSV 입력 결과 동일", np.allclose(same.value.to_numpy(), q.value.to_numpy(), equal_nan=True))

print("\n판정:", "전부 통과" if ok_all else "실패 있음", f"(임시 결과 폴더 {TMP})")
