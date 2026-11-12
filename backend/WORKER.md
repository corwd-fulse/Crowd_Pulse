# Vision 분석 Worker

## CPU/GPU 실행 옵션 (2026-11-12)

기존 명령은 계속 사용할 수 있습니다. `--device` 기본값은 `cpu`, `--imgsz` 기본값은 `1920`이며 모델은 기존처럼 `--model-path`로 명시합니다. CPU 환경을 GPU 패키지로 교체하지 않았습니다. GPU는 별도 `out/gpu-validation-20261112/.venv-gpu`에서 실행합니다.

```powershell
# backend/에서, ROI 저장 후 생성한 새 테스트 작업 UUID 입력
$jobId = Read-Host '새 테스트 분석 작업 job_id'
.\out\gpu-validation-20261112\.venv-gpu\Scripts\python.exe -m app.worker `
  --model-path .\out\gpu-validation-20261112\yolo11n.pt `
  --algorithm-version vision-gpu-options-20261112 `
  --device cuda:0 --imgsz 960 `
  --allow-test-analysis --once --job-id $jobId
```

이 명령은 실제 분석을 실행하므로 이번 구현 검증에서 실행하지 않았습니다. 완료 작업 ID는 재분석하지 않으며 queued인 새 작업이 필요합니다. 후보 ROI는 API의 `ALLOW_TEST_ANALYSIS=true`, 작업의 `analysis_mode=test`, Worker의 `--allow-test-analysis`가 모두 필요합니다. 운영 작업은 승인 ROI 정책을 그대로 따릅니다. 이 옵션들은 ROI를 자동 승인하거나 기존 작업 스냅샷을 변경하지 않습니다.

`--device`는 cpu, cuda:N, 단일 숫자 인덱스를 지원합니다. `0`/`cuda`는 cuda:0으로 정규화하며 auto/다중GPU 목록/음수 인덱스는 거부합니다. `--imgsz`는 양의32배수만 허용합니다. batch=1과 FP32(half=False)는 고정합니다. 모델/트래커 임계값·계산식·CSV 컬럼은 바꾸지 않았습니다.

CUDA가 없거나 인덱스가 없으면 해당 선점 작업을 CUDA_UNAVAILABLE/CUDA_DEVICE_UNAVAILABLE로 실패 처리하며 CPU로 자동 대체하지 않습니다. 실제 AutoBackend 내부 모델 파라미터와 Predictor/입력/출력 장치를 검증하므로 원본 모델 래퍼가 CPU에 남는 Ultralytics의 복사 동작 때문에 GPU를 CPU로 오기록하지 않습니다. 정밀도가 다르면 INFERENCE_PRECISION_MISMATCH로 실패합니다.

CUDA OOM은 CUDA_OUT_OF_MEMORY로 기록하고 자동 재시도하지 않습니다. CLI는 OOM 후 exit1로 종료하여 같은 설정으로 나머지 큐를 연속 실패시키지 않습니다. 모델/입력 크기를 검토한 뒤 명시적으로 재시작해야 합니다. CPU/GPU 모두 기존 동일한 로컬 파일 잠금과 PostgreSQL advisory lock을 사용하므로 프로세스를 여러 개 실행해도 YOLO 동시 실행 슬롯은1개입니다.

실행 메타데이터 settings에는 model_name/model_sha256, imgsz, requested_device/device, Predictor/모델 장치, device_verified, batch/half/precision, torch_version/cuda_version/cudnn_version을 저장합니다. model_version, 알고리즘 레이블, 원본영상 해시/FPS/프레임수, AI 소스 해시와 ROI/analysis_mode 기록도 유지합니다. 기존 테이블·마이그레이션·API 응답 구조는 변경하지 않았습니다.

### GPU 환경 및 검증 범위

PyTorch2.5.1+cu118/torchvision0.20.1+cu118/Ultralytics8.4.211/NumPy1.26.4/OpenCV4.10.0.84를 유지한 채 GPU 환경에만 SQLAlchemy2.1.8, psycopg3.3.8, pydantic-settings2.16.2, pandas3.0.6, lap0.5.13과 의존성을 설치했습니다. 기존 CPU .venv는 변경하지 않았습니다.

GTX1060 3GB에서 동일12프레임(0.48초)을 무손실FFV1 클립으로 만들고 픽셀 일치를 확인한 뒤, Worker.run_tracking의 실제 어댑터와 실제 run_zone_pipeline/validate_bundle을 실행했습니다. 큐 선점/복구/작업 상태 변경/결과 저장은 하지 않았으며 같은 배타 잠금만 사용했습니다.

- GPU 설정: YOLO11n/960/cuda:0/batch1/FP32. Predictor 및 내부 파라미터 장치 검증 통과.
- 프레임12개, 평균 추적 인원10.7, 최고15. CSV에서 tracking ID 생성 확인.
- CSV행: crowd_observations2 / flow2 / individual_directions107 / individual_zone_agg2 / bottleneck_scores0. 0.48초 표본에서 병목 빈 결과는 정상이며 5개 모두 스키마 검증 통과.
- 계산 경로 전체 약12.33초(콜드 초기화와 CPU Optical Flow/구역 분석 포함). 이전56.74FPS는 워밍업 후 YOLO predict만 측정한 값이므로 서로 같은 속도 지표가 아닙니다. 전체 영상 처리 시간이나 실시간 처리 성능으로 환산하지 않습니다.
- CPU 회귀 및 GPU 설정 전달/CUDA 미지원/장치 검증/OOM/잠금/메타데이터/기존 결과 API 테스트137개 통과. 실제 저장된 결과에 대한 프론트 API 연동2개도 통과.

증거: `out/gpu-worker-validation-20261112/smoke-result.json`, `execution_metadata.json`, `compute/tracker/`, `compute/pipeline/`, `smoke.py`. GPU 작업의 실제 DB 선점부터 저장까지의 종단 간 검증은 새 유효한 작업에 대한 별도 승인이 있을 때 수행합니다. 이번 검증에서는 전체 영상 추론/원본 CSV 변경/실행 중 Worker 종료를 하지 않았습니다.

## 실행 구조

FastAPI는 `POST /api/analysis/jobs`에서 durable `queued` 작업만 만든다.
독립 CLI 프로세스가 조회하여 `queued → running → succeeded/failed`로 전이한다.
API 시작/reload는 Worker를 시작하지 않으며 HTTP 요청에서 YOLO를 실행하지 않는다.
기존 5초 React 폴링과 기존 작업 생성/목록/상세 경로는 유지한다.

현재 구현은 **단일 호스트의 단일 YOLO 실행 슬롯**을 목표로 한다.
CLI에는 동시 실행 수를 늘리는 옵션이 없다. Windows/Unix 로컬 파일 잠금과 PostgreSQL
세션 advisory lock(키 19201112)을 함께 소유해야 시작할 수 있다. 파일 존재 자체가 잠금은 아니다.
잠금 파일을 수동 삭제하지 말고 Worker 프로세스의 정상 종료를 사용한다.
정상 네트워크 조건에서는 다른 DB Worker의 기동도 advisory lock으로 차단한다.
다중 호스트·네트워크 분할을 포함하는 고가용성 실행 보장은 이번 단일 호스트 구현 범위 밖이다.

DB 선점은 `FOR UPDATE SKIP LOCKED`, 상태 `queued`, 생성 순서로 수행한다.
실행 소유 UUID `worker_token`과 `running`을 commit하고 요청 세션을 재사용하지 않는다.
각 단계/작업은 독립 session을 사용하며 17분 추론 동안 작업 행 잠금이나 데이터 세션을 유지하지 않는다.
별도 advisory-lock 연결만 유지하고 watchdog가 확인한다.

Worker는 별도 단일 프로세스 안에서 직접 AI 함수를 실행한다. 하위 YOLO 프로세스가 없으므로
Worker 프로세스 강제 종료 후 독립적으로 살아남는 AI 자식 프로세스도 없다.
계산 중 DB 잠금 연결이 끊기면 watchdog가 Worker를 exit=75로 즉시 종료한다.
로컬 파일 잠금은 프로세스가 끝날 때까지 유지되어 같은 호스트의 중복 추론을 막는다.
watchdog 종료 자체가 지연되어도 슬롯을 반환하기 전에 fail-stop한다.

## 마이그레이션·환경

신규 `20261112_04`는 다음 nullable/기본값 컬럼만 추가한다.

- `videos.roi_snapshot`: 영상에 명시적으로 등록한 ROI.
- `analysis_jobs.roi_snapshot`: 작업 요청 시의 독립 스냅샷.
- `analysis_jobs.analysis_mode`: 기본 `production`; 명시적 `test` 구분.
- `analysis_jobs.worker_token`: Worker 소유권/실행 식별자.

기존 행·파일·이전 마이그레이션은 삭제/초기화하지 않는다. 기존 작업에는 production 기본값이
들어가며 ROI·token은 NULL이다. 기존 queued 작업에 임의 ROI를 소급 적용하지 않는다.
ROI가 없던 작업은 실행 전 명시적인 오류로 실패한다. ROI 등록 후 새 작업을 요청한다.
과거 실행·수동 import의 running 작업 중 token이 없는 것은 자동 복구 대상으로 삼지 않는다.

```powershell
# backend/에서, 기존 DB 환경 설정 사용
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m alembic check
```

모델은 **미리 확보한 로컬 파일**이 필요하다. Worker는 모델을 자동 다운로드하지 않는다.
Ultralytics/torch/OpenCV/pandas/numpy/lap 등 기존 AI 의존성을 설치해야 한다.
`YOLO_AUTOINSTALL=false`로 런타임 패키지 설치도 끈다.
기존 함수 계산식/임계값은 변경하지 않는다. adapter는 모델 경로 지정과 작업 간 모델/추적 캐시
리셋만 수행한다. 실제 모델 해시와 AI 소스 여섯 파일 해시를 실행 메타데이터에 보존한다.
실행 중 AI 소스가 바뀌면 결과를 정상 버전으로 오인해 저장하지 않고 실패 처리한다.

## ROI와 모드

```text
PUT /api/videos/{video_id}/roi  body: 기존 Zone 형식의 JSON snapshot
GET /api/videos/{video_id}/roi
```

필수 구조는 `video_resolution`, `status`, `approved_for_analysis`, `zones:[{name,polygon}]`다.
승인된 ROI는 `status=approved, approved_for_analysis=true`여야 한다.
후보는 candidate/candidate_unvalidated + false다. 등록 API가 파일을 자동 읽거나 승인하지 않는다.
서버에 전송한 승인 상태는 명시적인 검토자의 책임이며, JSON 스키마 검사가 품질 승인은 아니다.
실제 영상 디코딩 및 해상도 일치는 Worker가 실행 전에 검증한다.

작업 생성 시 영상의 등록 ROI를 deepcopy하여 저장한다. 이후 영상 ROI를 바꿔도 기존 작업의
좌표·승인 상태는 바뀌지 않는다. JSON 설정 파일 자체를 덮어쓰지 않는다.

운영 작업은 승인된 ROI만 실행한다. 후보 사용에는 **둘 다** 필요하다.

1. API 서버의 `ALLOW_TEST_ANALYSIS=true` + 요청 body `analysis_mode="test"`.
2. Worker CLI의 `--allow-test-analysis`.

운영 Worker는 test 작업을 선점하지 않는다. 테스트 허용 Worker도 production 작업에 후보 ROI를
사용할 수 없다. 결과/작업 API와 실행 스냅샷에 analysis_mode가 남아 운영/테스트를 구분한다.
방향 품질은 항상 `unvalidated`로 저장하며 성공/ROI 승인과 방향 품질 승인을 혼동하지 않는다.
기존 수동 CSV 가져오기 API는 계속 제공되지만 실행 중 Worker 소유 작업의 결과를 수동으로
덮어쓸 수 없다. 토큰과 상태를 다시 확인한 Worker만 그 작업의 신규 결과를 저장한다.

## 일반 Worker 실행

FastAPI와 다른 터미널에서 실행한다. 아래 algorithm-version은 실제 팀 릴리스/버전으로 지정한다.
이 명령은 계속 큐를 처리하므로 이번 개발 과정에서는 실행하지 않았다.

```powershell
.\.venv\Scripts\python.exe -m app.worker `
  --model-path .\out\vision-pipeline-validation-20261112\yolo11m.pt `
  --algorithm-version vision-baseline-v1
```

실행 순서:

1. 단일 슬롯 잠금 확보, 중단 작업 복구.
2. queued 작업 선점과 running/started_at/token commit.
3. DB 영상 경로를 uploads/videos 아래로 제한하고 파일 존재·영상 상태 확인.
4. 작업 ROI 스키마·승인·해상도·선택적 영상 해시 확인. 실제 FPS/프레임 수 확인.
5. UUID 기반 `uploads/analysis/{job_id}/{token}-.../`에 독립 출력 디렉터리 생성.
6. `track_video(source, tracker_dir, 'video')` 실행.
7. 반환된 detections 경로가 작업 디렉터리 안인지 확인. 원본 프레임 수와 처리 수 비교.
8. ROI 스냅샷에서 Zone 로드 후 `run_zone_pipeline()` 실행.
9. 출력 경로 검증, 실제 시작·종료 시각/설정/버전/해시 저장.
10. 기존 CSV 검증·저장 서비스에 worker_token과 함께 전달.
11. 서비스가 결과와 succeeded 상태를 같은 트랜잭션으로 commit.

헤더만 있는 detections를 곧바로 오류로 판단하지 않는다. 후속 파이프라인과 저장 스키마 검증의
결과로 정상 여부를 결정한다. 현재 평균 인원 CSV의 유입/유출 결측 등 출력이 저장 계약에
맞지 않는 경우에는 0으로 조작하지 않고 실패 처리한다. 빈 병목 CSV는 기존대로 정상 처리한다.

실패/중단에는 계산식이나 ROI를 보정해 재시도하지 않는다. 결과 저장 rollback 후 독립 session에서
token이 여전히 일치하는 running 작업만 failed로 바꾼다. 공개 error_message는 단계/안전한 오류
코드만 보존하며 DB 연결 파라미터나 임의 예외 문자열은 넣지 않는다.

출력: ROI 스냅샷, tracker CSV, pipeline CSV, `execution_metadata.json`, `worker.log`.
실패한 부분 출력도 보존하고 기존 출력·결과를 삭제/덮어쓰기하지 않는다.
최종 CSV 사본은 기존 저장 서비스의 `uploads/results/{job_id}/import-.../`에도 보관된다.

## 중단·복구 정책

- Ctrl+C/SIGTERM: 현재 작업을 중단하고 가능한 경우 failed와 종료 시각을 기록한다.
- 강제 종료/DB 연결 상실: running이 남을 수 있다. 다음 Worker가 단일 슬롯을 확보한 후
  worker_token이 있는 미완료 running 작업을 `WORKER_INTERRUPTED`로 failed 처리한다.
- 복구 시 finished_at은 실패를 확정한 시각이며 미완료 분석의 실제 완료 시각을 의미하지 않는다.
- **자동 재시도/동일 작업 재실행은 하지 않는다.** 재시도는 공식 API로 새 job을 만든다.
  기존 출력과 이력은 보존한다.
- 이미 결과가 있는 비정상 running 행은 자동 덮어쓰기/시각 생성으로 복구하지 않는다.
  정상 경로는 결과+상태를 한 트랜잭션으로 저장하므로 이런 상태는 별도 점검 대상이다.
- 결과 저장이 불가능한 DB 장애에는 failed 기록도 지연될 수 있다. DB 복구 후 Worker를 재시작한다.
- 큐 처리 프로세스의 OS 서비스화/자동 재기동은 이번 단계에 포함하지 않았다.

## 실제 AI 1회 통합 검증 절차 — 아직 실행하지 않음

현재 보행로 ROI는 candidate다. 운영으로 승인하지 말고 명시적 테스트 경로를 사용한다.
아래 `<등록된 영상 UUID>`는 실제 GET /api/videos 응답의 ID로 대체한다.

API 서버를 실행하는 터미널:

```powershell
$env:ALLOW_TEST_ANALYSIS = 'true'
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

다른 터미널에서 ROI 등록·작업 요청(기존 같은 영상의 활성 작업이 있으면 409이므로 먼저 상태 확인):

```powershell
$videoId = '<등록된 영상 UUID>'
$roi = Get-Content -Raw -Encoding utf8 .\configs\zones\walkway_candidate_1920x1080.json
Invoke-RestMethod -Method Put -Uri "http://127.0.0.1:8000/api/videos/$videoId/roi" `
  -ContentType 'application/json' -Body $roi
$body = @{video_id=$videoId; analysis_mode='test'} | ConvertTo-Json
$job = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/analysis/jobs' `
  -ContentType 'application/json' -Body $body

# 이 작업만 처리하고 종료. 다른 queued 작업을 소비/복구하지 않는다.
.\.venv\Scripts\python.exe -m app.worker --once --job-id $job.id `
  --allow-test-analysis `
  --model-path .\out\vision-pipeline-validation-20261112\yolo11m.pt `
  --algorithm-version vision-baseline-v1

Invoke-RestMethod "http://127.0.0.1:8000/api/analysis/jobs/$($job.id)"
Invoke-RestMethod "http://127.0.0.1:8000/api/analysis/jobs/$($job.id)/results"
```

CPU에서 이전 실행은 약 17분 47초였다. 이번에는 상태 폴링과 최종 DB/결과 조회까지 검증한다.
candidate/test/unvalidated 상태가 유지되는지 확인한다. 큐에는 등록되지만 Worker를 별도로
실행하지 않으면 queued에 머무는 것이 정상이다.

## API 호환성

- 공식 작업 생성·조회 응답에 analysis_mode가 추가된다. 기본 요청 body는 기존대로 video_id만 가능하다.
- 기존 React가 받는 상태 문자열은 그대로다.
- **변경:** `POST /api/videos/{video_id}/analyze`는 deprecated/410이다. 더 이상 동기 실행하지 않는다.
  기존 동기 `status=analyzed`/CSV 경로 응답을 쓰는 클라이언트는 공식 작업 생성+조회로 이전해야 한다.

## 검증

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m alembic check
.\.venv\Scripts\python.exe -m app.worker --help
```

Worker 테스트는 실제 PostgreSQL의 rollback fixture와 임시 디렉터리, Mock AI 함수만 사용한다.
선점/중복 기동 차단, 경로/승인/해상도/해시 검사, production/test 분리, AI 예외, commit 실패,
중단 복구, stale token 거부, 독립 출력·원본 보존, 기존 API 회귀를 포함한다.
watchdog 종료는 독립 테스트 하위 프로세스에서 exit=75를 확인한다(YOLO는 호출하지 않음).
테스트 Worker는 fixture 작업 ID로 범위를 제한해 실제 queued 작업을 소비하지 않는다.
