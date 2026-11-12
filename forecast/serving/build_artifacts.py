"""서빙에 필요한 작은 산출물을 학습 데이터에서 만든다: 동 순서, 동별 정규화 평균, 특징 순서, 평가 지표, LSTM 가중치 사본.
사용: python build_artifacts.py   (forecast/serving/artifacts/ 에 저장. LightGBM은 gzip(약 25MB)으로 넣고 manifest.json에 버전·해시를 기록)"""
import sys, json, shutil, gzip, hashlib
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
sf = F / "seeds_lstm.csv"  # 명세서: LSTM은 시드 3개 이상 평균 ± 표준편차로 보고
if sf.exists():
    sd = pd.read_csv(sf).drop_duplicates("seed", keep="last").sort_values("seed")
    metrics["lstm_seeds_test_2026-10"] = {"seeds": sd.seed.tolist(), "per_seed": sd[["seed", "MAE", "RMSE", "sMAPE", "MAE_surge"]].to_dict("records"),
                                          "mean_std": {c: [round(float(sd[c].mean()), 2), round(float(sd[c].std()), 2)] for c in ("MAE", "RMSE", "sMAPE", "MAE_surge")},
                                          "note": "36에폭 같은 설정, 시드만 다름. 서빙 모델은 시드 0. sMAPE는 %."}
(OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "lgbm_1h.txt.gz").write_bytes(gzip.compress((F / "lgbm_1h.txt").read_bytes(), 6))  # 63.5MB -> 약 25MB (GitHub 50MB 경고선 아래)
sha = lambda f: hashlib.sha256((OUT / f).read_bytes()).hexdigest()
manifest = {
    "task": "서울 행정동 생활인구 1시간 뒤 예측", "areas": len(w.columns), "lookback_h": 168,
    "train_period": "2025-11-01 ~ 2026-08-31", "val_period": "2026-09", "test_period": "2026-10",
    "scale": {"file": "scale.csv", "rule": "동별 학습 구간(~2026-08-31 23시) 평균 인구(명). 입력 / scale = 모델 입력 비율값", "sha256": sha("scale.csv")},
    "dong_order": {"file": "dong_order.json", "sha256": sha("dong_order.json")},
    "feature_cols": {"file": "feature_cols.json", "sha256": sha("feature_cols.json")},
    "models": {
        "seasonal_naive": {"version": "seasonal-naive-168h", "file": None, "rule": "y(T) = 실측(T-168시간)"},
        "lightgbm": {"version": "lightgbm-1h-v1", "file": "lgbm_1h.txt.gz", "sha256": sha("lgbm_1h.txt.gz"), "trained_at": "2026-11-11T14:47:00+09:00", "params": {"objective": "l1", "learning_rate": 0.05, "num_leaves": 127, "rounds": 4000}},
        "lstm": {"version": "lstm-1h-36ep-seed0", "file": "lstm_1h_36ep.pt", "sha256": sha("lstm_1h_36ep.pt"), "trained_at": "2026-11-12T15:27:00+09:00", "params": {"hidden": 96, "layers": 2, "epochs": 36, "lr": "cosine 2e-3 -> 1e-5", "seed": 0}},
    },
}
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(metrics, ensure_ascii=False, indent=2)); print("동", len(w.columns), "| 특징", len(cols), "|", sorted(x.name for x in OUT.iterdir()))
