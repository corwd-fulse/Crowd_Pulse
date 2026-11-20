# 몰림 예측 학습 (JuPedSim 시뮬레이션 → LightGBM)

실제 영상에는 몰림 사례가 거의 없어 보행 시뮬레이션으로 학습 데이터를 만든다.
여기서 만든 모델을 `forecast/serving/artifacts/minutes/`에 넣으면 `forecast/serving/minute_forecast.py`가 영상 CSV로 5·10·15분 뒤를 예측한다.
명령은 저장소 루트에서 실행한다.

## 순서

```bash
# 0. (이미 됨) 실제 보행 실험(2018 crowdqueue)에 맞춰 보행 파라미터 보정 -> data/crowdqueue/ga_best_v2.json
python -m forecast.bottleneck.calibrate_ga --tag v2

# 1. 시나리오 생성: 0~119 직선·병목, 120~179 T자, 180~239 한산. 각 1시간, 1초 기록 -> data/simcrowd/
python -m forecast.bottleneck.sim_dataset --n 240 --workers 7

# 2. 공지 있음/없음 테스트(정확도·경보) -> runs/<이름>/results.md
python -m forecast.bottleneck.sim_announce --out <날짜_이름> --hide 0.2 --shapes straight,bottleneck,tjunction

# 3. 서비스 모델 학습 -> runs/<이름>/models/*.txt, config.json
python -m forecast.bottleneck.sim_serving train --out <날짜_이름> --hide 0.2 --shapes straight,bottleneck,tjunction \
  --drop speed,speed_1m,speed_5m,width,exit_width,entry_width,area,cap,is_bottleneck,is_tjunction

# 4. 서비스에 반영
cp forecast/bottleneck/runs/<날짜_이름>/models/*.txt forecast/bottleneck/runs/<날짜_이름>/config.json forecast/serving/artifacts/minutes/
```

- 시나리오 설정(`data/simcrowd/scenarios.csv`)과 보정값은 저장소에 있다. 시뮬레이션 결과(약 72MB)는 1번으로 다시 만든다.
  같은 설정이면 같은 결과가 나온다. 7코어 기준 하루 가까이 걸린다.
- 실행할 때마다 `runs/<날짜_이름>/`에 결과를 남기고, 이미 있는 이름은 덮어쓰지 않는다. 모델 파일(`models/`)과 예측 parquet은 커밋하지 않는다.
- `--drop`: 속도와 절대 크기(폭·면적) 특징은 시뮬레이션과 실제 영상에서 뜻이 달라 서비스 모델에서 뺀다.
- 결과 보기: `streamlit run forecast/app_sim.py`

## 파일

| 파일 | 역할 |
|---|---|
| `crowdqueue.py`, `crowdqueue_lstm.py`, `calibrate_ga.py`, `snapshot_ga.py`, `sim_forecast_test.py` | 실제 실험 데이터 읽기, 보행 파라미터 GA 보정, 보정 확인 |
| `sim_dataset.py` | 시나리오 생성 (JuPedSim) |
| `sim_models.py` | 10초 칸, 특징, 정답(5·10·15분 뒤 인원 변화), 공지 일정 특징, LightGBM·LSTM 비교 |
| `sim_announce.py` | 공지 있음 / 일부 / 없음 조건별 정확도와 경보 |
| `sim_serving.py` | 서비스 모델 학습 |
| `test_bottleneck.py` | 테스트 |
