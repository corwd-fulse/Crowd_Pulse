# CROWD PULSE AI

### AI 기반 인파 밀집 분석 및 혼잡 예측 시스템

**CROWD PULSE AI**는 영상 분석과 시계열 예측 기술을 활용하여 인파의 밀집도와 이동 흐름을 분석하고, 위험 상황을 사전에 감지하기 위한 AI 기반 모니터링 시스템입니다.

단순히 인원수를 측정하는 데 그치지 않고, **인파 이동 패턴, 병목 구간, 혼잡도 변화 및 미래 혼잡 위험**까지 분석하는 것을 목표로 합니다.

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 프로젝트명 | CROWD PULSE AI |
| 개발 인원 | 3명 |
| 프로젝트 유형 | AI 기반 웹 모니터링 서비스 (포트폴리오·PoC) |
| 개발 방식 | GitHub Organization + Fork + Pull Request |
| 개발 상태 | 영상 분석 → DB 저장 → 대시보드 → 시계열 예측까지 통합 동작. 예측·경보는 실험적 지표 |

## 현재 구현 범위

| 기능 | 상태 | 비고 |
|---|---|---|
| 사람 탐지 (YOLO11m) · 추적 (ByteTrack) | 구현 | 구역(ROI)별 초 단위 인원, 유입·유출 |
| 밀집 인원 추정 (DM-Count) · 밀도 히트맵 | 구현 | YOLO와 독립된 보조 지표. "AI 모델 비교" 탭 |
| YOLO + DM-Count 보완 인원 | 구현 | YOLO 박스 밖 DM-Count 밀도를 더한 값 |
| 이동 흐름 (Optical Flow, 멀티 프로세스) | 구현 | 방향·속도, 역방향·체류 |
| 병목 점수 · 대기열 · 체류 시간 | 구현 | |
| 시계열 DB (PostgreSQL + TimescaleDB) | 구현 | 분석 결과·예측 결과 모두 DB 저장 |
| 통합 대시보드 (React) | 구현 | 기본 결과(현재 관측) / AI 모델 비교 / 시계열 예측(실험적 전망) |
| 장시간 영상 분석 재개 (체크포인트) | 구현 | 추적·Optical Flow·DM-Count 단계별 이어 하기 |
| 10·30·60초 뒤 인원 예측 (LightGBM, LSTM, Ridge) | 구현 · 실험적 | 분석이 끝나면 자동 실행(worker `--auto-forecast`) |
| 5·10·15분 뒤 예측 · 밀도 경보 | 구현 · 성능 미달 | 아래 "한계" 참고 |
| 이상 흐름 3단계 경보 | 구현 · 미검증 | 정답 라벨 없이 빈도 기준으로 조정 |
| 실제 물리 밀도(명/㎡) | 미구현 | 구역 면적은 가정값(50 px = 1 m) |

## 시스템 구성

```text
영상 업로드 (React) ──> FastAPI ──> PostgreSQL/TimescaleDB
                                       ^
GPU worker (원격 GPU 서버, SSH 터널) ───┘
  YOLO + ByteTrack → Optical Flow → 병목 점수 → (확장) DM-Count·체류·대기열
  → (--auto-forecast) 시계열 예측 → crowd_predictions / video_forecast_runs
```

## 기술 스택

| 구분 | 기술 |
|---|---|
| Frontend | React, TypeScript, Vite, Tailwind CSS, Recharts |
| Backend | Python, FastAPI, SQLAlchemy, Alembic |
| Database | PostgreSQL, TimescaleDB (hypertable) |
| Computer Vision | YOLO11, ByteTrack, OpenCV Optical Flow, DM-Count |
| Prediction AI | LightGBM, LSTM (PyTorch), Ridge AR, Isolation Forest |
| Infra | Docker Compose, 원격 GPU 서버(NVIDIA L40S) + SSH 역방향 터널 |
| Collaboration | Git, GitHub, Fork, Pull Request |

## 프로젝트 구조

```text
Crowd_Pulse/
├── frontend/       # React + Vite + TypeScript 대시보드
├── backend/        # FastAPI, 분석 worker, Alembic 마이그레이션 (문서: backend/*.md)
│   ├── app/        # worker.py, services/ (pipeline, optical_flow, density_estimation, job_forecast ...)
│   ├── routers/
│   └── alembic/
├── forecast/       # 시계열 예측·이상 흐름 패키지 (run_job_forecast, training/train_video)
└── docker-compose.yml
```

## 실행 방법

```bash
# 1) DB · 백엔드 · 프론트엔드 (로컬 worker 컨테이너는 켜지 않는다)
docker compose up -d db backend frontend
#    대시보드 http://localhost:5173 · API http://localhost:8000/docs

# 2) GPU worker (원격 GPU 서버에서, DB는 SSH 역방향 터널 15432 -> 5432)
venv/bin/python -m app.worker --model-path models/yolo11m.pt --algorithm-version <버전> --device cuda:0 \
  --enable-extended-analysis --pixels-per-meter 50 --flow-workers 16 \
  --dmcount-model-path models/dmcount_qnrf.pth --dmcount-gamma 1.5 --dmcount-input-scale 1.5 \
  --auto-forecast --forecast-python <lightgbm·torch가 설치된 파이썬>

# 3) 예측만 다시 돌리기 / 파일 결과를 DB로 옮기기
python -m app.job_forecast <job_id> <uploads/analysis/<job_id>/<run> 폴더>
python -m app.import_video_forecast --all
```

Windows에서 `db` 컨테이너가 `bind: ... 5432 ... forbidden` 으로 뜨지 않으면, Hyper-V(WinNAT)가 5432가 포함된
포트 범위를 예약한 것이다(`netsh interface ipv4 show excludedportrange protocol=tcp` 로 확인).
Hyper-V는 TCP 동적 포트 범위 안에서 예약하는데, 이 범위가 1024부터로 바뀌어 있으면 5432도 예약될 수 있다
(`netsh int ipv4 show dynamicport tcp`, Windows 기본값은 49152부터).
- 즉시 복구(관리자 PowerShell): `net stop winnat; net start winnat` 후 `docker compose up -d db`
- 재발 방지(관리자, 재부팅 후 적용): `netsh int ipv4 set dynamic tcp start=49152 num=16384`

`docker compose down -v` 는 DB 볼륨을 지우므로 쓰지 않는다.

## AI 모델 성능과 한계 (2026-11-26 기준, 영상 4개)

평가는 모두 **직전값(persistence, 지금 인원이 그대로 간다고 보는 단순 예측)** 대비 개선율(skill)로 했다.

**초 단위 예측 (10·30·60초 뒤)** — 각 영상을 학습에서 뺀 모델로 채점(처음 보는 영상 기준)

| | 10초 | 30초 | 60초 |
|---|---|---|---|
| LightGBM (미리 학습, 영상 4개) | +10.3% | +11.2% | +16.3% |
| LSTM (미리 학습, 영상 4개) | +7.8% | +14.4% | +17.1% |

- 4개 영상 모두에서 직전값보다 낫다. 이전 모델(42분 영상 1개로 학습)은 새 영상 3개에서 직전값보다 나빴다(30초 최대 -281%).
- 인원이 크게 바뀔 때(상위 20%) 늘지/줄지 방향은 60초 뒤 기준 약 86~92% 맞히지만, 변화 크기는 실제의 약 25~55%만
  따라간다(늦고 작게 예측). 이 분석은 영상 안에서 학습한 모델 기준이다.

**분 단위 예측 (5·10·15분 뒤)** — 대부분 직전값보다 오차가 크다. 학습 데이터(영상 약 3시간)가 부족하고
구역 면적·출구 방향이 가정값이기 때문으로 본다. 대시보드에 "실험적"·"직전값보다 나쁨"으로 표시한다.

**인원 측정** — 직접 센 프레임 9장(±2명 정도의 대략적 수작업)과 비교하면 YOLO는 대체로 실제와 비슷했고(한 장면은 약 10명 과다),
DM-Count(QNRF 학습)는 사람이 가까이 크게 찍힌 이번 영상들에서 크게 적게 셌다. DM-Count는 YOLO가 놓치는 멀리 있는 작은 사람을
보완하려고 넣은 모델이라 원거리 밀집 장면에 맞다. 카메라 상황에 따라 맞는 방식이 다르다.

**경보** — 규칙(급증·정체) + Isolation Forest + 병목 점수의 3단계 경보는 1분에 0.24~0.61건으로 조정했지만,
실제 사고·혼잡 정답 라벨로 검증하지 않았다. 대시보드에 "미검증"으로 표시한다.

## 향후 개선 방향

- 다른 장소·시간대 영상 추가 후 재학습 (`python -m forecast.training.train_video ...`)
- 정답 인원 라벨 프레임(50~100장) 구축으로 YOLO·DM-Count 정확도 정량화, 카메라별 기본 인원 출처 선택
- 카메라 보정(Homography)·현장 실측으로 실제 구역 면적과 밀도 산출
- 혼잡·정체 구간 정답 라벨로 경보 Precision/Recall 검증
- 검증 데이터가 충분해지면 잔차 분위수 기반 예측 오차 범위 표시

## 팀 구성

| 인원 | 담당 업무 |
|---|---|
| 팀원 1 | Full Stack (React, FastAPI, DB, AI 모델 연동) |
| 팀원 2 | Vision AI (YOLO, ByteTrack, Optical Flow) |
| 팀원 3 | 시계열 AI (혼잡도 예측 및 분석) |

## 개발 진행 현황

- [x] 프로젝트 기획 및 주제 선정
- [x] GitHub Organization 생성
- [x] React + Vite + TypeScript 환경 구축
- [x] 프론트엔드 대시보드 개발
- [x] FastAPI 백엔드 구축
- [x] PostgreSQL / TimescaleDB 연동
- [x] Vision AI 모델 개발 및 연동 (YOLO, ByteTrack, Optical Flow, DM-Count)
- [x] 시계열 AI 모델 개발 및 연동 (분석 후 자동 예측, DB 저장)
- [ ] 예측·경보 정확도 검증 (정답 라벨 구축)
- [ ] 배포

## 프로젝트 목표

인파 밀집 위험을 단순히 탐지하는 것을 넘어, 인파의 움직임과 변화 추세를 분석하여 **위험 상황을 더 빠르게 인식하고 예측할 수 있는 시스템**을 구현하는 것이 최종 목표입니다.

---

**CROWD PULSE AI — Detect the Flow, Predict the Risk.**
