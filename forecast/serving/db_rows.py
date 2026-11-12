"""파이프라인 결과·지표를 통합 개발 명세서의 DB 테이블 모양(areas, population_history, forecasts, model_registry, model_metrics)으로 바꾼다.
DB에 넣는 일(INSERT/COPY)은 백엔드가 하고, 여기서는 열 이름과 형식을 명세서에 맞춘 DataFrame만 만든다.
명세서 기준: forecasts PK=(issued_at, area_id, model_version), target_at = issued_at + 1h, value NULL=산출 불가."""
import json
from pathlib import Path
import pandas as pd

A = Path(__file__).resolve().parent / "artifacts"
MANIFEST = json.loads((A / "manifest.json").read_text(encoding="utf-8"))
SOURCE = "서울 생활인구(OA-23016) 행정동 시간대"


def areas_rows(dongs):
    """areas: 이름 매핑 파일이 없어 name에 행정동코드를 넣는다(NOT NULL). 동 이름이 생기면 교체."""
    return pd.DataFrame({"id": list(map(str, dongs)), "name": list(map(str, dongs)), "grid_type": "행정동", "meta": "{}"})


def population_rows(wide):
    """population_history: wide(index=시각, columns=행정동코드, 값=인구) -> (observed_at, area_id, estimate, source)"""
    long = wide.rename_axis("observed_at").reset_index().melt("observed_at", var_name="area_id", value_name="estimate").dropna()
    long["observed_at"] = pd.to_datetime(long.observed_at).dt.tz_localize("Asia/Seoul")
    long["area_id"], long["source"] = long.area_id.astype(str), SOURCE
    return long


def to_forecasts_table(pred):
    """forecasts: run_forecast()["predictions"] -> (issued_at, target_at, area_id, model_version, value).
    issued_at = last_observed_at(예측 기준 시각). 산출 불가(status != ok)는 value를 NULL(NaN)로 둔다. unknown_area 동은 areas FK가 없어 뺀다."""
    p = pred[pred.status != "unknown_area"]
    r = pd.DataFrame({"issued_at": p.last_observed_at, "target_at": p.target_at, "area_id": p.area_id.astype(str),
                      "model_version": p.model_version.astype(str), "value": p.value.where(p.status == "ok").astype("float32")})
    assert not r.duplicated(["issued_at", "area_id", "model_version"]).any()  # 명세서의 PK
    return r.reset_index(drop=True)


def registry_rows():
    """model_registry: manifest.json 기준. mlflow_run_id는 아직 없다."""
    fw = {"seasonal_naive": "baseline", "lightgbm": "lightgbm", "lstm": "pytorch-lstm"}
    return pd.DataFrame([{"model_version": m["version"], "task": "forecast", "framework": fw[k], "params": json.dumps(m.get("params", {"rule": m.get("rule")}), ensure_ascii=False),
                          "trained_at": pd.Timestamp(m["trained_at"]) if m.get("trained_at") else pd.NaT,
                          "artifact_path": f"forecast/serving/artifacts/{m['file']}" if m.get("file") else None, "mlflow_run_id": None}
                         for k, m in MANIFEST["models"].items()])


def metrics_rows():
    """model_metrics: metrics.json의 10월 테스트 지표 -> 모델·지표별 한 행. sMAPE 값은 % 단위."""
    m = json.loads((A / "metrics.json").read_text(encoding="utf-8"))["test_2026-10"]
    v = {k: x["version"] for k, x in MANIFEST["models"].items()}
    ver = {"Seasonal Naive": v["seasonal_naive"], "LightGBM": v["lightgbm"], "LSTM_36ep": v["lstm"]}
    rows = [{"model_version": ver[k], "dataset": "seoul_living_pop_dong_1h", "split": "test", "samples": 317688, "metric": {"MAE": "MAE", "RMSE": "RMSE", "sMAPE_pct": "sMAPE"}[n],
             "value": val, "test_from": pd.Timestamp("2026-10-01 00:00", tz="Asia/Seoul"), "test_to": pd.Timestamp("2026-10-31 23:00", tz="Asia/Seoul"), "hardware": "GTX 1060 3GB"}
            for k, d in m.items() for n, val in d.items()]
    return pd.DataFrame(rows)
