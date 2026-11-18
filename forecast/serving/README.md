# 예측 파이프라인 2종

| | A. 영상 분석 CSV 파이프라인 | B. 생활인구 파이프라인 |
|---|---|---|
| 함수 | `run_video_forecast(폴더)` | `run_forecast(CSV)` |
| 입력 | 영상 추출 결과(`crowd_observations.csv`, `flow.csv`) | 서울 생활인구 행정동 CSV |
| 출력 | 구역별 **10·30·60초 뒤 인원** 예측 + 이상 흐름 **3단계(낮음/주의/높음)** | 행정동별 **1시간 뒤 인구** 예측 |
| 학습 | 쌓인 영상 CSV로 **오프라인 미리 학습**(`forecast/training/train_video.py`) → 서비스는 바로 예측. 모델 파일이 없으면 영상 안에서 즉석 학습하는 모델로 대체 | 학습된 모델 사용(재학습 없음) |

## A. 영상 분석 CSV → 전처리 → 단기 예측 + 이상 흐름 3단계
```python
from forecast.serving import run_video_forecast
res = run_video_forecast("분석결과폴더", horizons=(10, 30, 60), base_time="2026-10-31 21:00",
                         detections_csv="track/xxx_detections.csv", zones=zones, fps=24)   # 뒤 셋은 역방향·체류 특징용(없어도 동작)
res["forecast_path"]   # 폴더/forecast_video.csv
res["anomaly_path"]    # 폴더/anomaly_events.csv (단계 이벤트 + 규칙 + Isolation Forest, 명세서 anomaly_events 형식 확장)
res["timeline_path"]   # 폴더/anomaly_timeline.csv (구역별 매초 단계와 신호)
res["info"]            # 영상 길이, 예측 시간·모델별 MAE와 직전값 대비 개선율, 단계별 초 수, IF·병목 반영 여부
```
CLI: `python -m forecast.serving.video_pipeline 분석결과폴더 --horizons 10 30 60 --target mean10 --base-time "2026-10-31 21:00"`

- 입력은 팀 영상 파이프라인(`run_zone_pipeline`)이 저장하는 폴더 그대로(`crowd_observations.csv` 필수, `flow.csv`·`bottleneck_scores.csv`·`individual_directions.csv`는 있으면 사용).
- **예측 대상(`target`):** 기본 `mean10` = 10초 평균 인원(탐지 잡음을 줄인 값, 명세서 `crowd_obs_10s`와 같은 집계). `raw` = 1초 원시 인원.
- **모델(걷기식, 매초 그 시각까지의 영상만 사용):** `persistence`(지금 값, 기준선) · `moving_avg_5s` · `ridge_ar`(최근 변화 + 유입·유출 + 다른 구역 유입, 강한 규제, 학습 행 30개 이상) · `lightgbm`(+ 다른 구역 인원 변화·이동량·역방향·체류, 30초마다 재학습, 학습 행 200개 이상 ≈ 4분 이상 영상).
- **미리 학습 모델(쌓인 영상으로 학습, 첫 초부터 예측):** `lightgbm_pre`(표 특징, 예측 시간별 1개) · `lstm_pre`(최근 60초 시퀀스, 10·30·60초를 한 모델로). 모델 파일이 없거나 그 예측 시간을 학습하지 않았으면 `status=no_pretrained_model`이고 나머지 모델은 그대로 동작한다. `pre_dir`(CLI `--pre-dir`)로 모델 폴더 지정, 기본 `artifacts/video/`.
- **기준선:** 생활인구는 Seasonal Naive(지난주 같은 시각)지만, 1초 단위 짧은 영상에는 요일·시각 주기가 없어 **Naive(`persistence`, 지금 값 그대로)** 를 기준선으로 쓴다.
- **평가:** `info["backtest"]`에 예측 시간·모델별 MAE·RMSE와 **직전값 대비 개선율**(skill = 1 − MAE/MAE(persistence), 0보다 크면 직전값보다 나음).
- **SHAP:** `lightgbm_pre` 행의 `shap_top` = 그 예측에 크게 기여한 특징 3개(LightGBM 내장 TreeSHAP, 사람 수 단위. 예: `d1 +2.61명, motion5 +2.41명, d2 +2.23명`). 변수 기여도 해석이며 원인을 뜻하지 않는다.

### 영상 모델 학습 (오프라인, `forecast/training/train_video.py`)
서비스는 예측만 하고, 학습은 영상 분석 결과가 쌓이면 따로 돌린다(서비스 요청 안에서 돌리지 않음).
```
python -m forecast.training.train_video 결과폴더들의_상위폴더 --horizons 10 30 60
```
1. **수집·전처리:** 하위 폴더의 `crowd_observations.csv`(+`flow.csv`)를 모두 찾아 서비스와 같은 `load_video_series`로 전처리. 형식이 틀린 폴더는 건너뛰고 `skipped_folders`에 남긴다.
2. **특징:** 서비스와 같은 `pre_design`. 카메라마다 인원 규모가 달라 "그 구역 지금까지 평균 인원"으로 나눈 비율로 학습한다(모든 카메라 공용 모델 1개).
3. **비교·평가:** 영상 단위로 나눠(GroupKFold) **처음 보는 영상**에서 Naive·LightGBM·LSTM의 MAE·RMSE·skill → `compare_video.csv`. 같은 영상을 앞뒤로 나누면 결과가 부풀려지므로 쓰지 않는다.
4. **SHAP:** 특징별 평균 기여(사람 수) → `manifest.json`의 `shap_importance`, `shap_importance.png`.
5. **교체:** LightGBM은 예측 시간별 skill > 0, LSTM은 평균 skill > 0일 때만 교체(`--force` 예외). 이전 파일은 `history/<이전 버전>/`에 보관. 서버는 파일 수정 시각을 보고 새 모델을 다시 읽으므로 재시작이 필요 없다.
- 최소 조건: 영상 2개, 학습 행 500개. LSTM은 GPU가 있으면 GPU, 없으면 CPU.

### 이상 흐름 3단계 (`forecast/anomaly_levels.py`)
구역별 매초 신호를 모아 단계를 정한다. 기준값은 **제안값**이며 혼잡·한산 라벨 구간이 생기면 튜닝한다.

| 단계 | 조건 (하나라도 해당) |
|---|---|
| **높음** | 병목 점수 ≥ 70 / 규칙 경보와 Isolation Forest 이상이 동시에 / "주의"가 20초 이상 연속 |
| **주의** | 병목 점수 40~70 / 규칙 경보(급증·정체) / Isolation Forest 점수 ≥ 0.8 |
| **낮음** | 그 외 |

- 흔들림 방지: 3초 이상 유지돼야 올리고, 5초 이상 풀려야 내린다. 병목 점수 기준은 팀 FR-ZON-05(40·70)와 같다.
- 규칙 경보: 급증 = 5초 전 대비 +30% 이상이면서 3명 이상 증가, 정체 = 이동량이 최근 5초 평균보다 50% 이상 감소(구역에 3명 이상 있을 때만). 3초 이상 이어질 때.
- Isolation Forest 특징 4개(명세서 MDL-08): `count_delta`, `motion`, `reverse_ratio`(구역 최근 10초 주 방향과 90° 넘게 다른 사람 비율), `dwell`(초당 몸 높이 0.1배 미만으로 거의 서 있는 사람 비율). 속도는 박스 높이로 나눠 카메라·fps와 무관하게 비교한다.
- 이벤트(`method=level`)마다 `level`(최고 단계), `duration_s`, `peak_sec`, `reason`(근거 문장 예: "5초 전 대비 인원 +4.6명(+57%) · 이동량 -88% · 정체 경보 + 급증 경보"), `features`(평소 대비 편차 z, 화면의 기여 막대용).
- 역방향·체류는 그 구역 평소보다 눈에 띄게 높을 때만 근거 문장에 넣는다(양방향 통행에서는 평소에도 역방향이 30~40%).
- 표현 규칙(FR-ANO-03): 참고 지표이며 "사고 확정", "위험 판정" 같은 표현은 쓰지 않는다.

결과 `forecast_video.csv` (UTF-8 BOM)
| 컬럼 | 타입 | 설명 |
|---|---|---|
| `zone_id` | 문자열 | 구역 이름(ROI) |
| `model` | 문자열 | `persistence` / `moving_avg_5s` / `ridge_ar` / `lightgbm` / `lightgbm_pre` / `lstm_pre` |
| `horizon_s` | 정수 | 몇 초 뒤 예측인지(10·30·60) |
| `target` | 문자열 | `mean10`(10초 평균 인원) / `raw` |
| `split` | 문자열 | `backtest`(실제와 비교) / `forecast`(영상 끝 + N초, 실제 없음) |
| `last_observed_sec`, `target_sec` | 정수(초) | 예측을 낸 시각, 예측 대상 시각 |
| `last_observed_at`, `target_at` | 시각 ISO 8601 | `base_time`(촬영 시작 시각)을 주면 채움 |
| `value`, `actual` | 실수(명) | 예측값, 실제(backtest만) |
| `status` | 문자열 | `ok` / `insufficient_history` / `gap_in_history` / `no_pretrained_model` |
| `shap_top` | 문자열 | `lightgbm_pre` 행만: 상위 기여 특징 3개와 기여(명). 다른 모델은 빈칸 |

`anomaly_timeline.csv`: `zone_id, sec, level, level_num, candidate, count, count_delta_5s, motion_drop, reverse_ratio, dwell, rule_kind, rule_strength, if_score, if_anomaly, bottleneck_score`

검증: `python -m forecast.serving.test_video_pipeline` (스키마, 미래 값 누수 없음(원시·10초 평균), 10분 합성 데이터 LightGBM, IF 윈도우 29/30, 단계 경계(병목 39/40/69/70), 흔들림 방지, 급증 주입 시 단계 상승, 근거 문장, 금지 표현, 타임라인 생성, 미리 학습: 영상 1개 거부·잘못된 폴더 건너뜀·비교표·SHAP 합 = 예측·처음 보는 영상에서 첫 초부터 예측·두 모델 누수 없음·모델 없음 표시·이전 버전 보관). Windows 콘솔에서는 `PYTHONIOENCODING=utf-8`을 켜고 실행

**한계:** 1분 남짓 영상은 비교 시점이 적어 예측 수치는 참고용이다(두 공개·팀 영상에서 학습 모델이 직전값을 확실히 이기지 못함). 예측을 개선하려면 같은 카메라의 긴 연속 영상 여러 개로 미리 학습하는 단계가 필요하다(계획 문서 참고).

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
