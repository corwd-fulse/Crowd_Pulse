# Optical Flow CPU 최적화 — 분석/구현/비교 리포트

40분 영상 CPU Optical Flow 분석이 5시간 이상 걸리는 문제에 대한 분석과, 정확도 손실 없는
최적화(Phase 1) + Checkpoint/Resume·진행률 기능(Phase 3)의 구현 결과, 그리고 해상도/FPS
조정의 속도-정확도 트레이드오프 비교 실험(Phase 2) 결과를 담는다. YOLO/ByteTrack 탐지·추적
결과와 CSV/API 스키마(`vision_csv.py`의 `SCHEMAS`)는 전혀 바꾸지 않았다.

## 병목 원인

`app/services/optical_flow.py`의 `cv2.calcOpticalFlowFarneback()` 호출(원본 해상도 dense
flow)이 전체 비용을 압도적으로 지배한다. 코드 레벨 오버헤드(zone mask 재비교, 프레임당 row
누적 후 pandas groupby)는 실측상 거의 영향이 없었다 — 아래 Phase 1 결과 참고. 즉 "불필요한
반복문/연산 제거"만으로는 이 문제를 해결할 수 없고, Farneback 자체의 호출 횟수(frame_stride)
또는 입력 해상도를 줄여야 체감 가능한 속도 개선이 나온다(Phase 2).

## Phase 1 — 정확도 손실 없는 최적화 (적용됨)

변경 내용 (`app/services/optical_flow.py`):
- zone mask를 매 계산 프레임마다 `mask==1`로 재비교하지 않고, 구역별 인덱스(`np.flatnonzero`)를
  한 번만 구해 재사용.
- 프레임×구역마다 row를 쌓아 마지막에 pandas `groupby().agg()`로 평균 내던 방식을, 루프 안에서
  (sec, zone)별 합계/개수를 직접 누적하는 방식으로 교체.
- 중복 정의된 `columns` 리스트 제거.

**검증**: 합성 영상에 대해 리팩터 전/후 구현을 같은 입력으로 돌려 비교한 결과, `sec`/`zone`
순서, `low_confidence`, `angle_deg`/`magnitude_px`/`motion`이 모두 float64 연산 순서 차이로
인한 잡음(최대 ~3e-16) 이내로 동일했다 — 이 저장소가 이미 "동치"로 인정하는 기준
(`test_hybrid_flow.py`의 `<2도, <0.1px`)보다 훨씬 작다. 기존 pytest 8종
(`test_optical_flow.py`, `test_extended_pipeline.py`) 전부 통과.

**실측 속도**: 640x360/90프레임(2종 zone), 960x540/150프레임(20종 zone) 두 규모에서 측정한
결과, 속도 차이는 측정 잡음 수준(0.95x~1.06x)이었다 — Farneback 자체가 전체 시간의
압도적 비중을 차지해서, 코드 레벨 정리는 체감 속도에 거의 영향을 주지 못한다. **Phase 1은
정확도 손실 없는 코드 정리로서 유지할 가치가 있지만, "5시간→?" 문제의 해법은 아니다.**

## Phase 3 — Checkpoint/Resume + 진행률/ETA (구현됨)

### Checkpoint/Resume
`compute_zone_motion()`에 `checkpoint_path`/`checkpoint_every_frames` 옵션 추가.
지정하면 주기적으로 `(frame_idx, 누적 acc, carry-forward 상태, 직전 grayscale 프레임)`을
디스크에 저장하고, 같은 `checkpoint_path`로 다시 호출되면(같은 영상/구역/설정 서명이 일치할
때만) 그 지점부터 `cv2.CAP_PROP_POS_FRAMES`로 seek해 이어서 계산한다. 정상 종료 시
checkpoint는 삭제된다.

**적용 범위를 명확히 제한했다**(합의한 대로): 이 재개는 **같은 프로세스·같은 job 호출 안**에서
Optical Flow 단계가 예외로 끊겼을 때 한 번 재시도하는 용도다(`app/services/pipeline.py`의
`_run_zone_pipeline`이 `compute_zone_motion` 예외를 잡아 checkpoint로 한 번만 재시도).
`worker.py`의 "동일 job 자동 재시도/재제출 금지" 정책(`WORKER.md`)이나 job 상태 큐 로직은
전혀 바꾸지 않았다 — 프로세스가 완전히 죽거나(강제종료) 컨테이너가 재시작되면 여전히
`recover_interrupted()`가 그 job을 `failed`로 마감하며, 새 job을 제출해야 한다.

**검증**: 루프 중간에 예외를 주입해 끊은 뒤 같은 checkpoint로 재개한 결과가, 끊김 없이 한 번에
실행한 결과와 완전히 동일함을 확인(`tests/test_optical_flow_checkpoint.py`,
`test_resumed_run_matches_single_uninterrupted_run`). 파이프라인 레벨 재시도 1회 동작도
별도 테스트로 검증(`test_pipeline_resumes_once_from_checkpoint_after_flow_exception`,
`test_pipeline_does_not_swallow_a_second_failure`).

### 진행률/ETA
`compute_zone_motion()`에 `progress_callback(frame_idx, frame_count)` 추가, 같은 주기로
호출된다. `worker.py`는 이를 받아 ① `uploads/analysis/<job_id>/flow_progress.json`
사이드카 파일에 쓰고, ② `analysis_jobs` 테이블의 새 컬럼(`progress_percent`,
`progress_eta_seconds`, `progress_updated_at`)을 가벼운 독립 세션으로 갱신한다. 진행률
보고 자체가 실패해도 분석 job은 절대 실패시키지 않는다(try/except로 격리).

DB 변경: `alembic/versions/20261120_12_add_analysis_job_progress.py` — 전부 nullable
추가 컬럼, 기존 행 백필 없음, 기존 패턴(`roi_snapshot`/`worker_token` 추가 방식)과 동일.
API: `routers/analysis.py`의 `AnalysisJobResponse`에 세 필드 노출 — 기존 5초 주기 polling
(`useAnalysisJobPolling.ts`)에 그대로 얹을 수 있다(이번 변경에는 프론트엔드 표시는
포함하지 않았다 — 백엔드/DB/API까지).

## Phase 2 — 해상도/FPS 비교 실험 (리포트만, 운영 코드에 반영하지 않음)

합의한 대로 **운영 코드(`flow_method`)에는 추가하지 않고, 비교 실험 결과만 남긴다.**
실제 샘플 영상(E05_001.avi)은 로컬에 없어(Drive에만 존재, `backend/samples/README.md`
참고) 합성 영상(1920x1080, 30fps, 60프레임, zone 6개, 블롭마다 다른 방향/속도로 움직이는
군중 흉내)으로 측정했다. **절대 시간/오차 값은 이 테스트 환경의 CPU와 합성 모션에 한정된
수치이며, 실제 CCTV 영상·실제 서버 사양에서는 다를 수 있다 — 상대적 경향(해상도↓/stride↑가
시간과 오차에 어떻게 영향을 주는지)만 참고할 것.**

| scale | stride | 60프레임 실측(s) | 프레임당(ms) | 각도오차(도, max) | 크기오차(px, max) | 40분 환산 추정 |
|---|---|---|---|---|---|---|
| 1.00 | 1 | 47.713 | 795.22 | 0.000 (기준) | 0.000 (기준) | 약 954분 |
| 1.00 | 2 | 24.406 | 406.76 | 3.323 | 0.0050 | 약 488분 |
| 1.00 | 4 | 11.133 | 185.54 | 13.711 | 0.0298 | 약 223분 |
| 1.00 | 8 | 5.146 | 85.77 | 113.969 | 0.0410 | 약 103분 |
| 0.75 | 1 | 22.220 | 370.34 | 2.902 | 0.0526 | 약 444분 |
| 0.75 | 2 | 11.199 | 186.64 | 4.689 | 0.0523 | 약 224분 |
| 0.75 | 4 | 6.153 | 102.55 | 17.257 | 0.0498 | 약 123분 |
| 0.75 | 8 | 3.223 | 53.72 | 170.031 | 0.0635 | 약 65분 |
| 0.50 | 1 | 10.778 | 179.63 | 2.781 | 0.1443 | 약 216분 |
| 0.50 | 2 | 4.804 | 80.07 | 3.837 | 0.1512 | 약 96분 |
| 0.50 | 4 | 2.805 | 46.75 | 16.057 | 0.1467 | 약 56분 |
| 0.50 | 8 | 1.466 | 24.44 | 124.707 | 0.1251 | 약 29분 |
| 0.33 | 1 | 4.734 | 78.90 | 9.038 | 0.3222 | 약 95분 |
| 0.33 | 2 | 2.827 | 47.12 | 10.514 | 0.3296 | 약 57분 |
| 0.33 | 4 | 1.373 | 22.89 | 19.953 | 0.3008 | 약 27분 |
| 0.33 | 8 | 0.746 | 12.43 | 114.918 | 0.2357 | 약 15분 |

기준(scale=1.0, stride=1)과 비교. 오차는 두 결과 모두 `low_confidence`가 아닌 (sec,zone)
행만 대상으로 한 `max(원형각도차)`/`max(|크기차|)` — `test_hybrid_flow.py`가 쓰는 것과 같은
방식.

**관찰**:
- 이 저장소가 기존에 "동치"로 인정하는 기준(`<2도, <0.1px`, `test_hybrid_flow.py`)을
  **완전히 만족하는 조합은 없다** — 가장 가까운 것은 scale=0.75·stride=1(2.90도, 0.053px)
  이지만 그래도 속도는 기준 대비 약 2.1배 개선에 불과하다.
- **stride를 키우면 각도 오차가 급격히 커진다**(특히 stride=8에서 모든 해상도에서
  100도 이상) — `optical_flow.py`의 frame_stride 독스트링이 이미 경고하는 "그 구간 동안
  등속이라는 가정" 문제가, 변위가 작을 때(이 합성 모션처럼 초당 수 px) 특히 심각하게
  드러난다. 작은 벡터는 노이즈에 방향이 아주 민감하기 때문이다.
- **해상도를 낮추면 크기(px) 오차가 해상도 축소 자체로 인해 바닥값이 생긴다**(scale=0.5에서
  stride와 무관하게 약 0.14px) — 보간/다운샘플링 손실로 보인다.
- 큰 속도 개선(5~10배 이상)은 전부 상당한 각도/크기 오차를 동반한다. **"정확도 손실 없이
  5시간을 몇 분으로 줄이는" 조합은 이 실험 범위에서 존재하지 않는다** — 속도와 정확도는
  뚜렷한 트레이드오프 관계이며, 어느 수준의 오차를 받아들일지는 실제 운영 영상으로
  재검증한 뒤 결정해야 한다.

**권장(결정은 아님)**: 실제로 큰 개선이 필요하다면, 정확도 영향이 비교적 작은
scale=0.75~1.0·stride=1~2 구간부터 실제 CCTV 샘플로 재검증하는 것을 다음 단계로 제안한다.
이번 라운드에서는 운영 코드에 반영하지 않았다.

## 바로 쓸 수 있는 손쉬운 레버

운영 코드 변경 없이 **이미 구현되어 있고 테스트된** `frame_stride` 옵션(worker.py
`--frame-stride`)을 현재 1보다 크게 설정하는 것만으로도(예: 2) 즉시 속도를 올릴 수 있다 —
단, 위 표에서 보듯 stride 자체도 정확도에 영향을 주므로 "정확도 손실 없음"으로 분류하지는
않았고, 실제 영상으로 재검증 후 사용을 권장한다.

## 추가 라운드 — Checkpoint 원자성 수정 + 서버 재시작 후 Resume + 실측 프로파일링

### Checkpoint 저장 형식 변경 (버그 수정)
기존에는 상태(json)와 직전 프레임(npy)을 **따로** `os.replace()`해서, 그 둘 사이에 프로세스가
죽으면 서로 안 맞는 checkpoint가 남을 수 있었다. 단일 파일(매직바이트+sha256 체크섬+pickle
페이로드)로 합쳐서 **한 번만** 원자적으로 교체하도록 바꿨다. 손상되거나(체크섬 불일치,
truncated, bad magic, unpicklable) 핑거프린트가 안 맞는 checkpoint는 구체적인 이유와 함께
무효 처리되고, 그 이전의 정상 checkpoint는 그대로 보존된다 — 실제로 저장 중간에 프로세스를
강제종료(`kill -9`에 해당)해서 확인했다(`tests/test_optical_flow_checkpoint.py`).

### 서버/컴퓨터 재시작 후 Resume
`worker.py`에 "resume manifest"(영상 해시·ROI·algorithm_version·frame_stride·flow_method·
imgsz·device·코드 해시·tracking 결과 경로를 담은 사이드카 파일)를 추가했다. 워커가 완전히
죽었다가(이 경우만 — 정상 종료/예외는 기존처럼 즉시 failed 처리된다) 재시작하면:
- `recover_interrupted()`가 유효한 resume manifest + checkpoint가 있는 job만 `queued`로
  되돌리고(소유권만 비움, `started_at`은 최초 시각 그대로 — job ID·시작 시각 연속성 유지),
  없거나 뭔가 안 맞으면(영상 교체/ROI 변경/코드 변경/파일 삭제/재시도 상한 초과 등) 기존처럼
  **구체적인 이유를 적어** `failed`로 마감한다(임의로 조용히 재개하지 않음).
- 재개된 job을 다음 `run_once()`가 집으면 YOLO/ByteTrack(tracking)은 **다시 돌리지 않고**
  이전 시도의 `detections.csv`를 그대로 재사용하며, Optical Flow만 checkpoint 지점부터
  이어서 계산한다. 결과는 기존과 동일하게 딱 한 번만 저장된다(중복 저장 불가).
- 재개 시도 횟수는 5회로 제한해서(`MAX_RESUME_ATTEMPTS`), 같은 지점에서 계속 죽는 영상이
  끝없이 재시도되지 않게 막았다.
- 여러 Worker가 동시에 재개를 시도할 수는 없다 — 기존에 이미 있던 전역 단일 실행 슬롯
  (로컬 파일 락 + Postgres advisory lock, `Worker.exclusive()`)이 `recover_interrupted()`/
  `run_once()` 양쪽 모두를 감싸고 있어서, 이번에 새로 만들 필요가 없었다.

### Seek 정확도 실측 (H.264, 실제와 비슷한 GOP 구조)
OpenCV `mp4v` 라이터로 만든 테스트 영상은 거의 매 프레임이 키프레임이라 참고가 안 돼서,
`ffmpeg libx264 -g 50 -bf 2`(GOP 50, B-frame 포함, closed GOP)로 다시 인코딩한 클립에서
GOP 경계 전후·중간 등 49곳을 "seek 후 읽기"와 "0번부터 순서대로 디코딩"으로 직접 픽셀
비교했다 — **49곳 전부 완전히 동일(0 mismatch)**. 이 OpenCV 빌드/컨테이너 조합에서는
`CAP_PROP_POS_FRAMES`가 프레임 정확했다. 단, 이건 "이 파일" 한정 결과이지 모든 코덱·컨테이너·
NVR 벤더에 대한 보증은 아니다 — `scripts/check_seek_accuracy.py --video <실제 영상>`으로
실제 CCTV 원본에 대해서도 재확인하는 걸 권장한다(합성 데모 모드도 있어 `--video` 없이
바로 실행 가능).

### 실측 프로파일링 (합성 1280x720/150프레임 클립, 이 샌드박스 CPU 기준)
`scripts/profile_optical_flow.py --video <클립> --zone "..."`로 디코딩/그레이스케일/
Farneback/구역집계 비용을 나눠서 재고, frame_stride 1 vs 2를 비교한다. 데모 실행 결과:

| 구분 | stride=1 | stride=2 |
|---|---|---|
| 처리 프레임 | 149 | 149 |
| Farneback 호출 수 | 149 | 74 |
| 총 시간 | 42.43s | 21.11s |
| decode | 0.16s (0.4%) | 0.18s (0.8%) |
| grayscale | 0.05s (0.1%) | 0.03s (0.1%) |
| **farneback** | **39.93s (94.1%)** | **19.76s (93.6%)** |
| aggregate(구역) | 2.29s (5.4%) | 1.14s (5.4%) |

Farneback이 전체의 **94%**를 차지한다는 걸 세 번째 독립적인 방법(이번엔 단계별 실측)으로도
다시 확인했다 — Phase 1(반복문/중복연산 제거)의 실측 개선이 거의 없었던 이유와 정확히
일치한다. **이 숫자는 이 샌드박스의 CPU와 합성 영상 기준이다.** 실제 영상·실제 서버에서
다시 돌려서 같은 비율이 나오는지 확인하는 걸 권장한다 — 특히 OpenCV 빌드의 SIMD/IPP 지원
여부(`cv2.getBuildInformation()`, 이 이미지는 AVX2/AVX512/Intel IPP가 이미 켜져 있음을
확인했다 — "최적화 안 된 빌드" 가설은 배제됨)와 실제 CPU 코어/클럭이 이 추정치를 바꾼다.

### 이번 라운드에서 확인한 것 / DB 없이 확인 못 한 것
- `_find_resumable_attempt`/`_write_resume_manifest`의 검증 로직(영상 해시·ROI·algorithm_
  version·frame_stride·flow_method·imgsz·device·코드 해시 불일치, 파일 누락, 재시도 상한,
  checkpoint 손상 — 요청하신 항목 전부)은 DB 없이 단위 테스트로 **직접 실행해 확인**했다
  (`tests/test_worker_resume.py`, 18개 전부 통과).
- `claim_next()`의 `started_at` 보존, `recover_interrupted()`의 큐/실패 분기, 전체
  재개→성공 흐름(`tests/test_worker.py::test_resume_after_restart_skips_tracking_and_
  preserves_job_identity`)은 실제 Postgres가 필요해서 이 샌드박스에서는 **실행하지 못했다**
  (다른 123개의 기존 DB 연동 테스트와 똑같이 `connection refused`로 에러난다 — 코드 버그가
  아니라 환경 제약임을 직접 확인). `docker compose`로 DB가 뜬 환경에서
  `pytest -m integration tests/test_worker.py`를 돌려 검증해 주시길 권장한다.
- YOLO/ByteTrack·`zone_analysis.py`·`bottleneck.py` 등은 이번에도 `git diff`로 전혀 손대지
  않았음을 재확인했다. 실제 작업 디렉터리(`C:\Users\Woori\crowd_pulse`)와
  `feature/vision-ai-extension`의 기존 미커밋 변경(`worker.py`의 sha256 구현 변경,
  `videos.py`의 업로드 용량 변경, 프런트엔드 4개 파일)은 건드리지 않았고, 별도 scratch
  worktree에서 병합 시뮬레이션까지 해서 충돌 없이 적용됨을 확인했다(시뮬레이션용
  worktree/브랜치는 확인 후 바로 삭제).

### 추가 확인 — 완전히 격리된 실제 Postgres+TimescaleDB로 통합 테스트까지 완료
공유 dev DB(`crowd-pulse-db`)는 전혀 건드리지 않고, **완전히 별도의 임시 Postgres+
TimescaleDB 컨테이너**(별도 compose 프로젝트, 자체 네트워크, 호스트 포트 미노출, 전용
볼륨 — 테스트 후 `down -v`로 흔적 없이 삭제)를 띄워 거기서 base부터 `20261120_12`까지
전체 마이그레이션을 적용하고 통합 테스트를 돌렸다. 기존 `crowd-pulse-backend`/`worker`/
`db`/`frontend` 컨테이너는 시작부터 끝까지 재시작 없이 그대로 떠 있었다(`docker ps`로
매 단계 재확인).

**결과: 312개 중 311개 통과.** 유일한 실패(`test_population_runtime.py`)는 수정 전
원본 코드에서도 동일하게 실패함을 이미 확인한, 완전히 무관한 기존 결함이다. 재개 관련
통합 테스트 3개 전부 실제 DB에서 통과:
- `test_resume_after_restart_skips_tracking_and_preserves_job_identity` — 서버 재시작 후
  같은 job_id·같은 started_at으로 checkpoint부터 재개, tracking 재실행 없음, 결과 저장.
- `test_resumed_success_stores_exactly_once_and_blocks_concurrent_worker` — **중복 DB
  저장 방지**(`store_result()` 호출 횟수를 직접 스파이해서 정확히 1회 확인) +
  **중복 Worker 실행 방지**(재개 작업 도중 두 번째 Worker의 `exclusive()`가
  `WorkerBusy`로 즉시 거부됨을 실제로 확인).
- `test_corrupted_checkpoint_after_crash_fails_with_clear_reason_not_silent_resume` —
  죽은 뒤 checkpoint 파일이 손상되면 임의로 재개하지 않고
  `WORKER_INTERRUPTED_NO_RESUME: RESUME_CHECKPOINT_INVALID (...)`로 명확히 `failed`.

이 과정에서 테스트 fixture의 실수(0-인덱스가 아닌 1-인덱스 프레임 번호 때문에 영상 길이
경계를 살짝 넘는 버킷이 생겨 `vision_csv`의 "offset outside video" 검증에 정당하게
걸림)를 발견해 고쳤다 — **테스트 fixture 문제였고, resume 로직 자체의 버그는 아니었다.**

**아직 확인하지 못한 것**: 공유 dev DB에는 harness 안전장치 때문에 마이그레이션을 적용하지
않았다 — 적용 여부는 승인 후 별도로 결정해야 한다. 실제 CCTV 원본에 대한 seek 정확도는
여전히 합성 H.264 클립으로만 확인했다(`scripts/check_seek_accuracy.py`) — 실제 영상이
있어야 마지막으로 재검증할 수 있다.
