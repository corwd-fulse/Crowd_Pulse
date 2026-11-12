"""서빙에 필요한 작은 산출물을 학습 데이터에서 만든다: 동 순서, 동별 정규화 평균, 특징 순서, 평가 지표, LSTM 가중치 사본.
사용: python build_artifacts.py   (forecast/serving/artifacts/ 에 저장. lgbm_1h.txt(60MB)는 복사하지 않는다)"""
import sys, json, shutil
from pathlib import Path
import numpy as np, pandas as pd
F = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(F))
from features_1h import load_features, load_wide
OUT = Path(__file__).resolve().parent / "artifacts"; OUT.mkdir(exist_ok=True)
w = load_wide(); w.columns = w.columns.astype(str)
d, cols, scale, _ = load_features(w)  # 학습 구간(~8월) 평균 = 정규화 평균
(OUT / "dong_order.json").write_text(json.dumps(list(w.columns), ensure_ascii=False), encoding="utf-8")
(OUT / "feature_cols.json").write_text(json.dumps(cols), encoding="utf-8")
scale.rename("scale").rename_axis("adm_cd").reset_index().to_csv(OUT / "scale.csv", index=False)
shutil.copy(F / "lstm_1h_36ep.pt", OUT / "lstm_1h_36ep.pt")
p = pd.read_parquet(F / "pred_test_1h.parquet"); y = p.actual.to_numpy()
def m(q):
    e = q - y; return {"MAE": round(float(np.abs(e).mean()), 2), "RMSE": round(float(np.sqrt((e ** 2).mean())), 2), "sMAPE_pct": round(float((2 * np.abs(e) / (np.abs(y) + np.abs(q))).mean() * 100), 3)}
roll = lambda f, mod: pd.read_csv(F / f, encoding="utf-8-sig").query("model == @mod")[["MAE", "RMSE", "sMAPE"]].mean().round(2).to_dict()
metrics = {"note": "인구 단위(명). test는 2026-10 전체 31만 7,688건(학습 ~2026-08, 검증 2026-09). LSTM은 36에폭 시드 0 한 모델, 롤링은 7~10월 4 fold 평균.",
           "test_2026-10": {"Seasonal Naive": m(p.naive.to_numpy()), "LightGBM": m(p.lightgbm.to_numpy()), "LSTM_36ep": m(p.lstm.to_numpy())},
           "rolling_mean_7_10": {"LightGBM": roll("rolling_lgbm.csv", "LightGBM"), "LSTM_36ep": roll("rolling_lstm.csv", "LSTM"),
                                 "Persistence": roll("rolling_lgbm.csv", "Persistence"), "Seasonal Naive": roll("rolling_lgbm.csv", "Seasonal Naive")}}
(OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(metrics, ensure_ascii=False, indent=2)); print("동", len(w.columns), "| 특징", len(cols), "|", sorted(x.name for x in OUT.iterdir()))
