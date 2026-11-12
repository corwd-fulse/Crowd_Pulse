"""서빙 함수(predict.py)가 학습 파이프라인이 저장한 10월 예측(pred_test_1h.parquet)과 같은 값을 내는지 확인한다.
사용: python test_parity.py   (living_pop_wide.parquet, lgbm_1h.txt, pred_test_1h.parquet 필요)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import Predictor
F = Path(__file__).resolve().parents[1]
w = pd.read_parquet(F.parent / "data/processed/living_pop_wide.parquet"); w.columns = w.columns.astype(str)
ref = pd.read_parquet(F / "pred_test_1h.parquet"); ref["adm_cd"] = ref.adm_cd.astype(str)
p = Predictor(lgbm_path=F / "lgbm_1h.txt", device="cpu")
times = ["2026-10-03 14:00", "2026-10-05 09:00", "2026-10-09 18:00", "2026-10-14 09:00", "2026-10-14 03:00", "2026-10-20 23:00", "2026-10-31 12:00", "2026-10-01 00:00"]
worst = {"lightgbm": 0.0, "lstm": 0.0}; rel = 0.0; mae = {"서빙(CPU fp32)": [], "저장된 예측(학습 때 GPU fp16)": []}
for t in times:
    T = pd.Timestamp(t); hist = w.loc[:T - pd.Timedelta(hours=1)].iloc[-200:]
    out = p.predict(hist, T).set_index("adm_cd"); r = ref[ref.datetime == T].set_index("adm_cd").reindex(out.index)
    d = {k: float(np.abs(out[k] - r[k]).max()) for k in worst}; worst = {k: max(worst[k], d[k]) for k in worst}
    rel = max(rel, float((np.abs(out.lstm - r.lstm) / r.actual).max()))
    mae["서빙(CPU fp32)"].append(float(np.abs(out.lstm - r.actual).mean())); mae["저장된 예측(학습 때 GPU fp16)"].append(float(np.abs(r.lstm - r.actual).mean()))
    print(f"{t}: 동 {len(out)}개 | 최대 절대 차이(명) LightGBM {d['lightgbm']:.4f} LSTM {d['lstm']:.3f} | 평균 예측 {out.lstm.mean():,.0f}명")
# 오류 처리 확인
for name, bad, tt in (("마지막 시각이 어긋남", w.loc[:"2026-10-14 05:00"].iloc[-200:], "2026-10-14 09:00"), ("이력 부족", w.loc[:"2026-10-14 08:00"].iloc[-50:], "2026-10-14 09:00")):
    try: p.predict(bad, tt); print("!! 오류가 나야 하는데 통과:", name)
    except ValueError as e: print(f"오류 처리 OK ({name}): {str(e)[:50]}")
print("LSTM 상대 최대 차이: %.4f%% | 실제값 대비 평균 오차(8개 시각 평균, 명): " % (rel * 100) + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in mae.items()))
# LightGBM: 저장된 값이 소수 1자리로 반올림돼 있어 0.05명까지는 같은 값. LSTM: 저장된 예측이 GPU 반정밀도(fp16) 출력이라 큰 동에서 수십 명 단위 잡음이 있다 -> fp16의 상대 정밀도(약 0.1%)의 몇 배 이내(0.3%)이고 오차가 나빠지지 않으면 통과
ok = worst["lightgbm"] <= 0.0501 and rel < 3e-3 and np.mean(mae["서빙(CPU fp32)"]) <= np.mean(mae["저장된 예측(학습 때 GPU fp16)"]) + 0.5
print("판정:", "통과" if ok else "실패", f"(최대 차이 {worst})")
