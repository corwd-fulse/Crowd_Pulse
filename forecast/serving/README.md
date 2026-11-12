# 예측 파이프라인 2종

| | A. 영상 분석 CSV 파이프라인 | B. 생활인구 파이프라인 |
|---|---|---|
| 함수 | `run_video_forecast(폴더)` | `run_forecast(CSV)` |
| 입력 | 영상 추출 결과(`crowd_observations.csv`, `flow.csv`) | 서울 생활인구 행정동 CSV |
| 출력 | 구역별 **N초 뒤 인원** 예측 + 이상 흐름 이벤트 | 행정동별 **1시간 뒤 인구** 예측 |
| 학습 | 미리 학습된 모델 없음. 학습이 필요한 모델은 **들어온 영상의 앞 60%로 그 자리에서 학습** | 학습된 모델 사용(재학습 없음) |

## A. 영상 분석 CSV → 전처리 → 단기 예측 + 이상 흐름 탐지
```python
from forecast.serving import run_video_forecast
res = run_video_forecast("분석결과폴더", horizon_s=10, base_time="2026-10-31 21:00")   # 폴더 또는 crowd_observations.csv 경로
res["forecast_path"]   # 폴더/forecast_video.csv
res["anomaly_path"]    # 폴더/anomaly_events.csv (명세서 anomaly_events 형식)
res["info"]            # 영상 길이, 구역, 백테스트 MAE, 모델별 status, Isolation Forest 실행 여부
```
CLI: `python -m forecast.serving.video_pipeline 분석결과폴더 --horizon 10 --base-time "2026-10-31 21:00"`

- 입력은 팀 영상 파이프라인(`backend/app/services/pipeline.py`의 `run_zone_pipeline`)이 저장하는 폴더 그대로. `flow.csv`가 없으면 이동량 없이 진행하고 Isolation Forest는 건너뛴다.
- 전처리: 구역별로 1초 간격으로 맞추고, 인원·이동량은 2초 이하 빈칸만 보간, 유입·유출 빈칸은 0.
- 모델: `persistence`(직전 값, 기준선) · `moving_avg_5s` · `ridge_ar`(학습 행 20개 이상) · `lightgbm`(학습 행 200개 이상 ≈ 4분 이상 영상). 모자라면 `insufficient_history`.
- 이상 흐름: 규칙 기반 경보 + Isolation Forest(구역별 윈도우 30개 이상일 때, 명세서 TC-ANO-04).
- **한계:** 1분 영상은 백테스트가 구역당 15개 남짓이라 MAE는 참고용이다. 1시간 뒤 같은 긴 예측은 같은 카메라의 며칠 분량 연속 영상과 별도 학습이 필요하다.

결과 `forecast_video.csv` (UTF-8 BOM)
| 컬럼 | 타입 | 설명 |
|---|---|---|
| `zone_id` | 문자열 | 구역 이름(ROI) |
| `model` | 문자열 | `persistence` / `moving_avg_5s` / `ridge_ar` / `lightgbm` |
| `split` | 문자열 | `backtest`(영상 뒤 40%에서 실제와 비교) / `forecast`(마지막 시각 + N초, 실제 없음) |
| `last_observed_sec`, `target_sec` | 정수(초) | 영상 기준 마지막 관측 시각, 예측 대상 시각 |
| `last_observed_at`, `target_at` | 시각 ISO 8601 | `base_time`(촬영 시작 시각)을 주면 채움 |
| `value`, `actual` | 실수(명) | 예측값, 실제 인원(backtest만) |
| `status` | 문자열 | `ok` / `insufficient_history` / `gap_in_history` |

검증: `python -m forecast.serving.test_video_pipeline` (스키마, 미래 값 누수 없음, 10분 합성 데이터로 LightGBM 학습, Isolation Forest 윈도우 29/30 경계, 짧은 영상, 잘못된 CSV)

---

# B. 생활인구 1시간 뒤 예측 파이프라인 (Seasonal Naive · LightGBM · LSTM)

서울 생활인구 CSV를 넣으면 전처리 → 3개 모델 예측 → 결과 CSV까지 한 번에 실행한다. **재학습 없이** `artifacts/`의 학습된 모델을 쓴다. CPU에서 동작한다(LSTM은 `map_location` 적용).

> 이 예측은 **생활인구 추정치(행정동 단위)** 용이다. 영상 CSV를 넣으면 A로 안내하는 오류가 난다.

## 1. 호출

### Python (FastAPI 등에서)
저장소 루트를 `sys.path`(또는 `PYTHONPATH`)에 두고:
```python
from forecast.serving import run_forecast

# 기본: 데이터의 마지막 시각 + 1시간 한 시점 예측 (운용용)
res = run_forecast("250_LOCAL_RESD_ADMDONG_202610")            # 폴더, CSV 파일, ZIP, 또는 이들의 리스트
res["path"]          # 결과 CSV 경로 (out_csv 생략 시 입력 옆에 forecast_YYYYMMDDHH_YYYYMMDDHH.csv)
res["predictions"]   # 결과 DataFrame (아래 스키마)
res["info"]          # 요약: 실측 기간, 예측 대상 구간, 입력 동 수, 보간·결측 칸 수, 모델별 status 건수, 모델 버전

# 구간: start~end의 매시각을 '그 1시간 전까지의 실측'으로 예측 (실측 vs 예측 비교 화면용, actual 채워짐)
res = run_forecast(["250_LOCAL_RESD_ADMDONG_202609", "250_LOCAL_RESD_ADMDONG_202610"],
                   out_csv="forecast_oct.csv", start="2026-10-01 00:00", end="2026-10-31 23:00")

# 모델 선택
res = run_forecast(path, models=("lightgbm", "lstm"))
```
- 모델은 처음 한 번만 읽고 캐시한다(약 1초). 같은 프로세스에서 반복 호출해도 다시 읽지 않는다.
- 구간 예측은 1시각당 약 0.3초(CPU, 427개 동 × 3모델).

### CLI (저장소 루트에서)
```
python -m forecast.serving.pipeline 250_LOCAL_RESD_ADMDONG_202610 --out result.csv
python -m forecast.serving.pipeline 250_LOCAL_RESD_ADMDONG_202609 250_LOCAL_RESD_ADMDONG_202610 --start "2026-10-01 00:00" --end "2026-10-31 23:00" --out oct.csv
```

### FastAPI 연동 예 (참고)
```python
from fastapi import FastAPI, HTTPException
from forecast.serving import run_forecast

app = FastAPI()

@app.post("/api/forecasts/run")
def run(csv_path: str):
    try:
        res = run_forecast(csv_path)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(422, str(e))
    df = res["predictions"]
    return {"path": res["path"], "rows": len(df), "last_observed_at": str(df.last_observed_at.max()),
            "items": df.astype({"last_observed_at": str, "target_at": str}).to_dict("records")}
```

## 2. 입력
| 형식 | 설명 |
|---|---|
| 서울 생활인구 원본 CSV | 서울 열린데이터광장 행정동 시간대 생활인구(OA-23016). 일별·월별 파일, 폴더, ZIP 모두 가능. 앞 4열(날짜, 시간, 행정동코드, 총생활인구)을 위치로 읽고 cp949/UTF-8, `;` 구분자 파일도 처리 |
| `population_history` 형식 CSV | DB에서 뽑은 `observed_at, area_id, estimate` (시간대 포함 시 한국 시각으로 변환) |

- **최소 168시간(7일)의 연속 이력**이 있어야 LightGBM·LSTM이 예측한다(169시간 이상 권장). 일별 파일이면 최소 8일치.
- 전처리 규칙은 학습 때와 같다(`forecast/data_pipeline/living_pop_io.py`): 같은 시각·동이 겹치면 마지막 값, 1시간 간격으로 맞춘 뒤 **2시간 이하 결측만 선형보간**.

## 3. 결과 CSV 스키마 (UTF-8 BOM, 모델당 한 행)
| 컬럼 | 타입 | 설명 |
|---|---|---|
| `area_id` | 문자열 | 행정동 코드 8자리 |
| `model` | 문자열 | `seasonal_naive` / `lightgbm` / `lstm` |
| `model_version` | 문자열 | `seasonal-naive-168h` / `lightgbm-1h-v1` / `lstm-1h-36ep-seed0` |
| `last_observed_at` | 시각 ISO 8601 (`2026-10-31T23:00:00+09:00`) | 이 예측에 쓴 실측의 마지막 시각 (= 명세서 `forecasts.issued_at`) |
| `target_at` | 시각 ISO 8601 | 예측 대상 시각 = `last_observed_at + 1시간` |
| `value` | 실수(명, 소수 1자리) | 예측값. 산출 불가면 빈 값 |
| `actual` | 실수 또는 빈 값 | 대상 시각의 실측(구간 예측에서 있을 때만) |
| `status` | 문자열 | `ok` / `insufficient_history`(이력 168시간 미만) / `gap_in_history`(최근 168시간 중간에 3시간 이상 결측) / `unknown_area`(학습한 427개 동이 아님. Seasonal Naive는 계산함) |

DataFrame(`res["predictions"]`)에서는 시각 컬럼이 `datetime64[ns, Asia/Seoul]`, 문자열 컬럼이 `string` 타입이다.

## 4. DB 연동 (`db_rows.py`)
| 명세서 테이블 | 함수 |
|---|---|
| `forecasts` | `to_forecasts_table(res["predictions"])` → `issued_at, target_at, area_id, model_version, value` (산출 불가는 NULL, `unknown_area` 제외, PK 중복 검사) |
| `population_history` | `population_rows(wide)` (`load_history(path)`로 얻은 wide) |
| `areas` | `areas_rows(dongs)` (동 이름 매핑이 없어 name에 코드) |
| `model_registry` | `registry_rows()` (`artifacts/manifest.json` 기준) |
| `model_metrics` | `metrics_rows()` (10월 테스트 MAE·RMSE·sMAPE) |

> 주의: 명세서의 `forecasts`는 `target_at` 기준 hypertable인데 PK에 `target_at`이 없다. TimescaleDB는 유니크 인덱스에 파티션 컬럼이 들어가야 하므로 테이블 생성 시 PK에 `target_at`을 추가해야 할 가능성이 높다(실제 DB에서 확인 필요).

## 5. 추론 파일 (`artifacts/`, 모두 Git 포함)
| 파일 | 내용 |
|---|---|
| `lgbm_1h.txt.gz` | LightGBM 모델 (gzip, 24.6MB. 원본 63.5MB) |
| `lstm_1h_36ep.pt` | LSTM 가중치 (36에폭, 시드 0) |
| `scale.csv` | 동별 정규화 평균(명) = 학습 구간(~2026-08-31) 평균. 모델 입력 = 인구 / scale |
| `dong_order.json` | 행정동 427개 순서 (LSTM 동 임베딩 인덱스) |
| `feature_cols.json` | LightGBM 입력 특징 25개 순서 |
| `manifest.json` | 모델 버전, 학습 구간, 파일별 sha256, 학습 일시·파라미터 |
| `metrics.json` | 평가 지표 (10월 테스트, 롤링 7~10월 평균, LSTM 시드별) |

`build_artifacts.py`가 학습 데이터에서 이 파일들을 다시 만든다.

## 6. 검증 (저장소 루트에서)
- `python -m forecast.serving.test_pipeline`: 원본 CSV(9·10월 폴더)로 30개 시각을 예측해 학습 때 저장한 예측과 비교(Seasonal Naive·실측 0.1명 이내, LightGBM 일치, LSTM 상대 0.13% 이내·실측 대비 MAE 동일), 스키마, 기본 모드(1,281행), 예외(이력 부족·모르는 동·3시간 결측), 입력 형식 동등성.
- `python -m forecast.serving.test_parity`: 배포 모델(gzip)로 8개 시각 재추론 일치 확인.
- 두 테스트에는 원본 CSV 폴더와 `forecast/pred_test_1h.parquet`, `data/processed/living_pop_wide.parquet`가 필요하다(용량 때문에 Git 미포함).
- LSTM은 학습 때 저장한 예측이 GPU 반정밀도 출력이라 CPU 서빙 값과 큰 동에서 최대 약 20명 차이가 난다. 실측 대비 오차는 같다.

## 7. 한계
- 1시간 뒤만 예측한다(여러 시간 앞은 모델이 지원하지 않음).
- 정규화 평균은 학습 구간 값으로 고정한다. 다른 해처럼 인구 규모가 크게 달라지면 LightGBM이 영향을 받는다(2024-06 데이터: MAE 283, 최근 7일 평균으로 기준값 보정 시 226).
- 공휴일은 `holidays`(KR) 라이브러리 + 2026-09-28 수동 추가에 의존한다.
- 427개 동 밖이나 코드가 바뀐 동은 LightGBM·LSTM이 예측하지 않는다(재학습 필요).
