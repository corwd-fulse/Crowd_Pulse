# 영상 구역 인원 예측 모델 — 입력(X)·정답(y)·컬럼 정리

2026-11-18 기준. 광화문 43분 영상 실험(`C:\Users\Woori\data-work\gwanghwamun\`)과 서비스 코드(`forecast/serving/video_pipeline.py`)에서 쓰는 정의.
행 하나 = **구역 하나 × 시각 t 하나**. 모든 특징은 **시각 t까지의 값만** 쓴다(미래 값 누수 없음).

---

## 1. 입력 CSV (팀 Vision 파이프라인 결과)

| 파일 | 컬럼 | 쓰임 |
|---|---|---|
| `crowd_observations.csv` | `sec`, `zone`, `count`, `inflow`, `outflow` | 구역별 초당 인원·유입·유출 (필수) |
| `flow.csv` | `sec`, `zone`, `motion` (+ `angle_deg`, `magnitude_px`, `low_confidence`) | 이동량(optical flow). `motion`만 사용 |
| `*_detections.csv` | `frame`, `t_ms`, `track_id`, `x1`, `y1`, `x2`, `y2`, `conf` | 사람별 박스. 도착 예정(A)·역방향·체류 계산 |
| `individual_directions.csv` | `frame`, `t_ms`, `track_id`, `angle_deg`, `speed_px`, `reliable` | 사람별 방향·속도. 역방향·체류 계산 |

구역 배정 기준: 박스 하단 중앙(발 위치)이 구역 다각형 안.

---

## 2. y (정답)

- 기본 대상 **`mean10`**: `y(t)` = 최근 10초 평균 인원 = mean(`count`[t−9 … t])
- 예측 시간 **H = 10, 30, 60초**
- 학습 정답(미리 학습·같은 조건 비교 모델):

  ```
  Y = ( y(t+H) − y(t) ) / ref(t)
  ref(t) = 그 구역의 시작~t 평균 인원 (최소 1)   ← 카메라마다 인원 규모가 달라 비율로 학습
  예측 인원 = y(t) + 모델 출력 × ref(t)
  ```

- 걷기식(서비스) `ridge_ar`·`lightgbm`은 비율 없이 `y(t+H) − y(t)`(명)를 바로 학습한다.

---

## 3. X — 기본 특징 12개 (`PRE_FEATURES`)

`lightgbm_pre`, 같은 조건 비교의 `lightgbm`이 쓰는 표 특징.

| 컬럼 | 정의 | ÷ ref |
|---|---|:-:|
| `level_ratio` | y(t), 지금 10초 평균 인원 | ✓ |
| `d1` ~ `d4` | y(t−k) − y(t), 1~4초 전과의 차이 | ✓ |
| `in5` | 최근 5초 유입 합 | ✓ |
| `out5` | 최근 5초 유출 합 | ✓ |
| `oth_in5` | 다른 구역들의 최근 5초 유입 합 | ✓ |
| `oth_dcnt5` | 다른 구역 인원 합의 5초 변화 | ✓ |
| `motion5` | 최근 5초 평균 이동량 | |
| `reverse_ratio` | 최근 5초 역방향 비율 | |
| `dwell` | 최근 5초 거의 서 있는 사람 비율 | |

> 광화문 실행은 `detections_csv`를 넘기지 않아 `reverse_ratio`·`dwell` = 0.

---

## 4. X — 실험에서 추가한 특징

### B. 여러 시간 단위 (10개, `ridge_multi`에 사용)

| 컬럼 | 정의 | ÷ ref |
|---|---|:-:|
| y − 30초 평균, y − 60초 평균 | 최근 평균 대비 지금 | ✓ |
| y(t) − y(t−30), y(t) − y(t−60) | 30·60초 추세 | ✓ |
| in30, in60, out30, out60 | 30·60초 유입·유출 합 | ✓ |
| 다른 구역 30초 변화 | 다른 구역 인원 합의 30초 변화 | ✓ |
| motion30 | 30초 평균 이동량 | |

### A. 도착·이탈 예정 (9개, `forecast/anomaly_levels.py` `arrival_features`, 모두 ÷ ref)

| 컬럼 | 정의 |
|---|---|
| `arr10`, `arr30`, `arr60` | 지금 구역 **밖**인데 H초 안에 **들어올** 사람 수 (최근 2초 속도로 직선 이동 가정, 1초씩 밀어 봄) |
| `dep10`, `dep30`, `dep60` | 지금 구역 **안**인데 H초 안에 **나갈** 사람 수 |
| `arrH − depH` | H초 안 순증 예정 |

이력 1초 미만이거나 느린(몸 높이 기준 0.2배/초 미만) 사람은 제자리로 본다.
선택 옵션(기본 꺼짐, 광화문에서 효과 없음): `stitch_gap_s`(끊긴 추적 잇기), `flow_map`(흐름 지도), `births_in/out{60,120}`(새로 나타난 사람 수).

---

## 5. 모델별 X·y

| 모델 | X | y |
|---|---|---|
| `persistence` (직전값) | 없음 (예측 = y(t)) | — |
| `moving_avg_5s` | 최근 5초 값 평균 | — |
| `ridge_ar` (걷기식, 서비스) | y(t), d1~d4, in5, out5, oth_in5 → **8개**, 원래 단위 | y(t+H) − y(t) |
| `lightgbm` (걷기식, 서비스) | y(t), d1~d4, in5, out5, oth_in5, oth_dcnt5, motion5, reverse_ratio, dwell → **12개**, 원래 단위 | y(t+H) − y(t) |
| `lightgbm_pre` / 같은 조건 `lightgbm` | `PRE_FEATURES` **12개** | 비율 Y |
| **LightGBM(D) + A** (현재 최고) | 12 + A 9 = **21개** | 비율 Y |
| `ridge_multi` + A | 12 + B 10 + A 9 = **31개** | 비율 Y |
| `lstm_pre` (1초 LSTM) | 시퀀스 **60초 × 8채널** (`SEQ_FEATURES`): `valid`(영상 시작 전 0), `level_ratio`·`in5`·`out5`·`oth_in5`·`oth_dcnt5`(÷ ref), `motion5`, `rel_to_now`(각 시점 − 지금 수준) | 비율 Y, 10·30·60초 **3개 동시** |
| 10초 칸 LSTM (1분 경로) | 10초 칸 **12개(2분) × 6채널**: 인원·유입·유출·다른 구역 인원(÷ ref), 이동량, 지금 칸 대비 변화 | 다음 10초 칸 **6개**(10~60초 뒤) |

LightGBM(D) 설정: 잎 7개, 잎당 최소 80행, 특징 60%만 사용, 학습률 0.03, 나무 400개, L1 손실.

---

## 6. 출력 `forecast_video.csv`

| 컬럼 | 뜻 |
|---|---|
| `zone_id` | 구역 |
| `model` | 모델 이름 |
| `horizon_s` | 예측 시간(초) |
| `target` | `mean10` 또는 `raw` |
| `split` | `backtest`(실제 있음) / `forecast`(영상 끝 + H초) |
| `last_observed_sec`, `target_sec` | 예측을 낸 시각, 예측 대상 시각 |
| `last_observed_at`, `target_at` | 위 두 시각의 실제 시각(촬영 시작 시각을 주면) |
| `value` | 예측 인원 |
| `actual` | 실제 인원(backtest만) |
| `status` | `ok` / `insufficient_history` / `no_pretrained_model` |
| `shap_top` | `lightgbm_pre` 예측에 크게 기여한 특징 3개 |

---

## 7. 같은 조건 비교 결과 요약 (광화문, 뒤 40%, 직전값 대비 개선율)

| 모델 | 10초 뒤 | 30초 뒤 | 60초 뒤 |
|---|---|---|---|
| ridge (12개) | +5.1% | +8.5% | +8.8% |
| LightGBM(D) (12개) | +9.7% | +8.6% | +9.9% |
| LightGBM(D) + A | **+13.3%** | +12.8% | **+11.7%** |
| ridge_multi + A | +11.1% | **+14.3%** | +11.6% |
| LSTM (1초, 시드 0) | +1.8% | +8.7% | +9.8% |
| LightGBM(D) + A + LSTM 평균 | +10.3% | +13.4% | +11.7% |

한계: 영상 1개(같은 카메라·같은 날)를 시간으로 나눈 비교, 사람이 적고 병목 장면 없음. B·A 특징은 아직 서비스 코드에 연결하지 않았다.
