# Forecast 전용 실행 환경

이 환경은 저장된 서울 생활인구 LightGBM/LSTM의 **실행 검증 환경**이다.
학습 당시 환경을 재현했다고 주장하지 않는다. 원본 `forecast/`, 기존 requirements, CPU/GPU 환경은 보존한다.

## 현재 구성과 실행

- Python: `C:\Users\Woori\Desktop\Crowd_Pulse\.venv-forecast\Scripts\python.exe`
- Python 3.11.17 / Windows x64 / CPU 전용. `include-system-site-packages=false`.
- 직접 의존성: `requirements-forecast-runtime.txt`
- 전이 의존성까지 고정: `requirements-forecast-runtime.lock.txt`
- 기존 `requirements-forecast.txt`는 조사 당시의 비고정 선언으로 그대로 남겨 두었다.

| 패키지 | 검증한 버전 |
|---|---|
| torch | 2.15.0+cpu |
| lightgbm | 4.7.0 |
| numpy | 2.4.6 |
| pandas | 3.1.1 |
| scikit-learn | 1.9.1 |
| holidays | 0.108 |
| pyarrow | 26.0.1 |

lock은 Windows x64/Python 3.11용 CPU wheel URL을 포함한다. 플랫폼 공통 lock이나 학습 원본 lock이 아니다.
PyTorch는 [공식 CPU 설치 안내](https://pytorch.org/get-started/locally/)의 배포 서버를 사용했고,
LightGBM은 [공식 설치 안내](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html)에 따른 pip binary wheel을 설치했다.
`pip check`를 통과했으며 소스 빌드나 CUDA 설치는 하지 않았다.

이미 구성된 이 PC에서는 환경을 새로 만들 필요가 없다. 다른 Windows/Python 3.11 PC에서 처음 준비할 때는
빈 전용 venv에만 `python -m pip install -r backend/requirements-forecast-runtime.lock.txt`를 실행한다.
이 파일을 기존 영상 분석 CPU/GPU 환경에 설치하지 않는다.

API의 인터프리터 선택 우선순위:

1. 프로세스 환경 변수 `CROWD_FORECAST_PYTHON`
2. `backend/forecast.runtime.local`의 `python` (상대 경로는 backend 기준)
3. API 프로세스의 Python (기존 동작)

현재 local 파일은 `../.venv-forecast/Scripts/python.exe`를 가리킨다. Git에는 포함하지 않는다.
다른 PC에서는 `forecast.runtime.example.json`을 참고하여 local 파일을 작성한다.
기존 `.env`, Conda, CPU/GPU venv의 PATH/패키지를 변경하지 않는다.
기존 API가 구버전 코드를 계속 실행 중이면 API 코드 리로드가 필요하다. 영상 분석 Worker를 종료할 필요는 없다.

## API 사용

- GET `/api/population-forecast/catalog`: 모델 버전, 지원 코드, 의존성 상태, 실제 자식 프로세스의 로드 상태.
- POST `/api/population-forecast/upload`: CSV/ZIP multipart 파일.
- POST `/api/population-forecast/predict`: `{"observations":[...]}` JSON.

```json
{
  "observations": [
    {"observed_at": "2026-10-01T00:00:00+09:00", "area_id": "11110515", "estimate": 10000}
  ]
}
```

위 1행은 형식 예시이며 실행하면 이력 부족으로 null을 반환한다. 모델 검증용 합성 CSV는
`validation/forecast_runtime_20261117/SYNTHETIC_interface_only.csv`에 명시적으로 구분해 두었다.
실제 데이터처럼 해석하거나 정확도 측정에 사용하지 않는다.

최소 168시간의 연속 이력, 169시간 권장. `observed_at`은 정시, `area_id`는 8자리 문자열,
`estimate`는 음수가 아닌 수치 또는 명시적 결측이다. 시각에 시간대가 없으면 KST로 해석한다.
API는 최대 100,000행, 427개 지역, 파일 10MiB를 받는다. ZIP은 압축 해제 합계 32MiB/CSV 128개 상한이다.
427개 지역 전체를 요청할 때 169시간은 72,163행이다. 월 전체를 그대로 넣으면 행 제한을 넘을 수 있다.

대상 시각은 **마지막 입력 시각 + 1시간**이다. 입력이 과거 자료이면 예측 대상도 과거일 수 있다.
현재 벽시계의 1시간 후를 자동으로 예측하는 실시간 서비스가 아니다.
응답은 지역×모델별 `area_id, model, model_version, last_observed_at, target_at, value, status`와 비교 정보이다.
공급하지 않은 지역의 값을 임의로 반환하지 않는다. 부족/잘못된 이력은 null/상태 또는 422이며 0명으로 대체하지 않는다.

`catalog.status=ready`는 아티팩트 해시/모듈 탐색을 통과하여 시도할 수 있다는 뜻이다.
`model_state`는 `not_loaded/loading/loaded/failed`를 구분한다. 실제 첫 추론 시 별도 프로세스에서 로드한다.
모델 사용 실패는 503, 프로세스 추론 대기 제한 90초는 504, API 대기 제한은 95초이다.
최대 4개 진행/대기 요청과 8개 결과 캐시를 유지한다. 과부하는 429이다.
라이브러리나 모델을 바꾼 후에는 API 프로세스/캐시를 갱신해야 한다.

자식 프로세스에만 CPU 장치와 OMP/MKL/OpenBLAS 한 스레드 제한을 적용한다.
기존 영상 Worker를 사용하거나 종료하지 않는다. timeout 정리는 해당 API가 생성한 Forecast 자식에만 적용한다.

## 실제 입력과 특징

기준: `forecast/serving/pipeline.py:load_history/run_forecast`, `predict.py:Predictor`.
서울 427개 행정동의 생활인구 수를 예측하며, 행사장 A/B/C·위험 점수·이동 경로 모델이 아니다.

LightGBM의 25개 특징은 `feature_cols.json` 순서와 실제 Booster.feature_name이 일치한다.
12개 인구 lag(1/2/3/6/12/24/48/72/96/120/144/168h), 차이 2개, 평균/표준편차 4개,
지난주 대비 비율, 공휴일 3개, hour/dow/dong을 사용한다. 행정동별 scale로 나눠 입력하고 다시 인구 단위로 복원한다.

LSTM은 행정동마다 `(168,8)`의 인구 비율+달력 입력, 목표 시각 달력 7개, 행정동 embedding ID,
마지막 인구 비율을 사용한다. 2층 hidden=96, embedding=16, 137,521개 파라미터의 state_dict를 CPU에 로드했다.
행정동 순서와 scale은 `dong_order.json`, `scale.csv`에 고정되어 있다.

Seasonal Naive는 `pipeline._naive`의 168시간 계절 기준선이며 학습 가중치가 없다.
현재 PopulationForecast API는 LightGBM/LSTM 두 모델을 반환한다.
`forecast/lstm_1h.pt`는 별도 파일이며 serving 기본값은 manifest에 지정된 `artifacts/lstm_1h_36ep.pt`이다.

## 촬영 세션 CSV와 video_pipeline 비교

| 항목 | 촬영 세션 학습 CSV | 기존 video_pipeline |
|---|---|---|
| 시간 | 완료 구간 끝 `video_offset_ms`/검증된 timestamp | 1초 시작 `sec` |
| 집계 | 1/5/10/30/60초 선택 | 1초로 reindex; 단위 변경 모델 아님 |
| 인원 | `person_count` | `count` |
| 구역 | venue/session/camera/job/ROI/target 식별자 | `zone` 한 컬럼 |
| 유입/유출 | inflow/outflow, 결측 유지 | 같은 이름; 일부 결측을 0으로 채우는 기존 처리 있음 |
| 이동량 | motion, 저신뢰이면 null | 별도 flow.csv에서 motion을 merge; low_confidence는 읽지 않음 |
| 방향 | 보정 전 direction/reverse 필드는 null | 방향 특징을 직접 사용하지 않음 |
| 결측 | 0 대체·보간 없음 | count/motion 내부 최대 2칸 보간, 유입/유출 일부 0 채움 |
| 여러 카메라 | long-format, 자동 합산 없음 | 카메라 차원 없음; 단순 병합하면 같은 zone이 충돌 |
| 예측 | 학습 특징만, 미래 정답 미생성 | persistence, moving_avg_5s, ridge_ar, lightgbm |
| LSTM | 연동 계약 준비 | 영상용 LSTM 구현/고정 가중치 없음 |
| horizon | export 자체는 horizon 없음 | 양의 정수 초; 기본 10초, 30/60도 호출 가능 |

기존 `run_video_forecast`는 영상의 앞 60%에서 ridge/LightGBM을 실행 중 학습하고 뒤 40%를 평가한다.
이번 작업에서는 이를 호출하거나 `anomaly_video`의 IsolationForest를 fit하지 않았다.
IsolationForest는 10초 sliding 특징의 유효 창이 구역당 30개 이상이고 motion이 완전할 때 실행되는 사후 이상 탐지이다.

영상이 확보되면 팀원에게 전달할 순서:

1. 촬영 세션·시각·영상·분석/ROI·관측 구역·목표 구역을 확인하고 **1초 간격** CSV와 스키마를 추출한다.
2. 장소/날짜/세션별로 먼저 학습/검증/테스트를 분리한다. 동시 촬영 카메라는 서로 다른 split으로 나누지 않는다.
3. 중복 시야를 합산하지 않는다. 카메라/ROI별 입력을 따로 만들고 `zone`을 전역 유일 식별자로 매핑한다.
4. `sec=video_offset_ms/1000-1`, `count=person_count`, inflow/outflow를 crowd_observations.csv로 대응시킨다.
   motion은 동일한 sec/zone의 flow.csv로 대응한다. 5초 데이터를 1초 데이터로 이름만 바꾸지 않는다.
5. 저신뢰·누락·중복·불연속 구간은 별도로 표시하고 유효한 연속 구간을 선택한다. 관측되지 않은 값을 0으로 만들지 않는다.
6. 기존 loader의 보간/0 채움이 이 정책과 다르므로, 결측이 있는 데이터를 바로 기존 함수에 넣지 않는다.
   수정이 필요하면 담당자가 별도 전처리/평가 정책을 승인하여 진행한다. 원본 코드를 이번 작업에서 바꾸지 않았다.
7. 모델별 최소치와 horizon을 확인한 뒤 담당자가 학습한다. 현재 작업에서 자동 학습을 실행하지 않는다.

유효 결측 없는 N개 1초 샘플에서 훈련 창 수는 `floor(0.6*N)-H-4`이다.
ridge는 20창, LightGBM은 200창이 필요하다. H=10/30/60초의 이론적 최소 N은
ridge 57/90/140초, LightGBM 357/390/440초이다. 이는 코드상 하한이며 운영 성능 보장이 아니다.
30분 영상도 가림/누락으로 유효 창이 부족할 수 있다. 여러 날짜·장소와 독립 정답으로 검증해야 한다.
목표 집결 구역을 예측하려면 해당 구역의 미래 정답도 필요하며 접근 카메라 인원을 정답으로 바꾸지 않는다.

## 검증 재실행

전용 Python으로 `validation/forecast_runtime_20261117/benchmark.py`를 실행하면 합성 입력과 저장 모델 실행 기록만 생성한다.
기존 CPU API Python으로 `validation/forecast_runtime_20261117/api_check.py`를 실행하면 FastAPI와 별도 Forecast 자식을 검증한다.
둘 다 기존 DB나 원본 CSV에 결과를 쓰지 않는다. 생성 파일은 해당 validation 폴더에만 둔다.
실제 생활인구/참조 Parquet 부재로 원래 테스트 기간의 MAE/RMSE 및 학습 당시 출력 parity는 재검증하지 못했다.
