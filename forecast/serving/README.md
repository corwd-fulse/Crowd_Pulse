# 서빙용 1시간 뒤 생활인구 예측 (LightGBM + LSTM 36에폭)

FastAPI 연동용 최소 구성. CPU에서 동작한다(`map_location` 적용 완료).

## 파일
| 파일 | 내용 | Git |
|---|---|---|
| `predict.py` | `Predictor.predict(history, target_time)` — 실제 예측 함수 | 포함 |
| `artifacts/dong_order.json` | 행정동 427개 순서 (LSTM 동 임베딩 인덱스) | 포함 |
| `artifacts/scale.csv` | 동별 정규화 평균(명, 학습 구간 ~2026-08 평균) | 포함 |
| `artifacts/feature_cols.json` | LightGBM 입력 특징 25개의 순서 | 포함 |
| `artifacts/metrics.json` | 평가 결과(MAE, RMSE, sMAPE) | 포함 |
| `artifacts/lstm_1h_36ep.pt` | LSTM 가중치 (554KB) | 포함 |
| `artifacts/lgbm_1h.txt` | LightGBM 모델 (60MB) | **별도 전달** (GitHub 100MB 한도 이내지만 50MB 초과 경고) |
| `data/processed/living_pop_wide.parquet` | 행정동 시간별 인구 이력 (25MB) | **별도 전달** (원천 공개 데이터에서 재생성 가능) |
| `build_artifacts.py`, `test_parity.py` | 산출물 생성, 저장된 예측과 일치 검증 | 포함 |

## 사용
```python
from predict import Predictor
p = Predictor(lgbm_path="lgbm_1h.txt")
out = p.predict(history, "2026-10-14 09:00")   # DataFrame[adm_cd, target_time, lightgbm, lstm] (명)
```
`history`: index=시각(1시간 간격, 마지막 행이 T-1시간), columns=행정동코드(문자열), 값=생활인구(명). 최소 168행(권장 169행), 최근 168행에 결측 없음. 조건을 어기면 `ValueError`.

## 검증 (`python test_parity.py`)
- 2026-10의 8개 시각에서 학습 파이프라인이 저장한 예측과 비교. LightGBM은 소수 1자리 반올림 범위(0.05명) 안에서 일치.
- LSTM은 저장된 예측이 GPU 반정밀도 출력이라 큰 동에서 최대 약 20명 차이가 나지만(상대 0.13% 이내), 실제값 대비 오차는 같은 수준(290.48 대 290.55).

## 알아둘 점
- 동 427개 외 지역/동 코드가 바뀐 경우는 지원하지 않는다(LSTM 동 임베딩이 학습한 427개 기준).
- 정규화 평균은 학습 때 값을 고정해 쓴다. 이력의 규모가 크게 달라지면(예: 다른 해) LightGBM이 영향을 받는다 (2024-06 테스트: 평균 보정 전 MAE 283, 보정 후 226).
- 공휴일은 `holidays` 라이브러리(KR) + 2026-09-28 수동 추가. 라이브러리 버전이 달라지면 공휴일 판정이 바뀔 수 있다.
- LSTM은 36에폭 시드 0 한 모델이다. 명세서의 시드 3개 평균 ± 표준편차 보고는 진행 중이다.
