# 새 영상 분석 흐름과 현재 차단 요소 (2026-11-12)

분석 요청 버튼은 queued 작업을 등록합니다. FastAPI 실행은 Worker 실행이 아닙니다. 실행 중인 Worker가 없으면 대기 상태가 유지됩니다. ROI를 그리는 작업도 추론을 시작하지 않습니다. 결과 조회만으로 새 탐지/인원 데이터를 생성하지 않습니다.

## 실제 확인 상태

최신 업로드의 서로 다른 밀집 영상은 `KakaoTalk_261112_183759.mp4`, video_id `ce0a7d1e-a771-4f3a-9844-c15a6c4e281a`입니다. 1280×720, 24 FPS, 1437프레임, 59.875초, 14,753,385바이트입니다. 업로드 DB 행/실제 파일/원본 스트리밍을 확인했습니다. 사용자의 기존 브라우저 탭 내부 상태를 직접 읽지는 않았으며, API 목록과 별도 Chrome 탭에서 이 영상을 선택해 검증했습니다.

연결 작업 `3d215ce9-33f2-4f89-9bd1-520dde4a159e`는 production/queued, started_at=null, worker_token=null입니다. 영상의 서버 ROI도 없고 작업의 요청 당시 ROI 스냅샷도 없습니다. 해당 작업 출력 폴더/탐지 CSV가 없습니다. 이 영상에 그린 구역이 화면에 보이더라도 서버 ROI가 저장됐다는 의미가 아닙니다. 실행 중인 app.worker CLI는 없었으며 Torch 2.14.1+cpu/CUDA 비활성 환경입니다. 모델 `backend/out/vision-pipeline-validation-20261112/yolo11m.pt`는 40,684,120바이트로 존재합니다. 실제 추론은 실행하지 않았습니다.

현재 API의 ALLOW_TEST_ANALYSIS는 false입니다. Worker를 켜는 것만으로 이 기존 작업을 성공시킬 수 없습니다. Worker preflight는 요청 당시 ROI 누락으로 ROI_NOT_CONFIGURED_OR_INVALID를 반환합니다. 영상에 ROI를 나중에 저장해도 과거 작업 스냅샷은 수정되지 않습니다. 현재 활성 작업의 중복 방지 정책 때문에 같은 video_id의 새 작업 요청도 차단됩니다. 취소 API/ROI 승인/ROI 버전 API는 이 작업에서 임의로 추가하지 않았습니다. 기존 작업의 상태도 변경하지 않았습니다.

## 코드 변경

- 영상 관리에서 분석 상태·영상/작업 ID·분석 요청/모드 선택을 원본 미리보기 위에 배치합니다. 미분석, 대기, AI 실행, 완료, 실패 원인을 구분하고 Worker가 별도로 필요함을 안내합니다. 기존 5초 폴링을 유지합니다.
- 요청 성공 응답을 즉시 목록에 반영하고 그 작업을 결과 선택 대상으로 예약합니다. 성공 전에는 그 선택을 다른 완료 작업으로 대체하지 않습니다. 폴링에서 succeeded가 확인되면 탐지/관측 결과를 자동 조회합니다.
- 작업 응답의 읽기 전용 roi_configured는 요청 당시 스냅샷 유효성만 표시합니다. ROI 본문은 응답에서 제외합니다. 현재 영상 ROI를 저장했다고 과거 작업을 실행 가능하다고 오인하지 않습니다.
- GET /api/analysis/capabilities는 현재 서버의 테스트 허용 및 외부 Worker 실행 방식을 반환합니다. 테스트 모드는 서버에서 허용한 경우에만 UI에서 선택할 수 있습니다. 설정을 자동 변경하지 않습니다.
- 메인 대시보드 업로드의 분석 요청도 저장된 승인 ROI를 먼저 조회하여 ROI 없는 production 작업을 UI에서 추가 생성하지 않도록 막습니다. 직접 API 호출 계약을 강제로 바꾸지는 않았습니다.
- VideoZoneCounts는 기존 결과 summary/zone-metrics GET을 재사용합니다. 페이지 전체를 읽고 job_id/video_id를 검사하여 영상 재생의 floor(currentTime) 초에 해당하는 평균 인원을 표시합니다. 0과 결측을 구분하며 현재 프레임 박스 수/고유 인원 수로 표시하지 않습니다. 집계는 분석 당시 ROI이며 편집 초안에 맞춰 재계산하지 않습니다. 미완료 작업에는 조회/표시하지 않습니다.

## 실제로 새 분석을 시작하는 절차 (이번에는 실행하지 않음)

현재 작업을 임의 변경하지 않는 경로는 동일 원본 파일을 별도 영상으로 다시 업로드하여 새 video_id를 얻는 것입니다. 기존 영상/작업/CSV는 보존합니다. 기존 영상 그대로 진행하려면 막혀 있는 대기 작업의 명시적 취소·재요청 기능을 별도로 설계/승인해야 합니다.

1. 새로 업로드한 영상을 영상 관리에서 선택합니다. 분석 요청 전에 해당 영상 해상도에서 ROI를 작성합니다.
2. 미리보기/좌표 검증 후 후보 ROI를 최초 저장합니다. 저장된 후보와 미저장 초안을 구분합니다.
3. 운영 분석은 별도 검토로 승인된 ROI가 필요합니다. UI는 자동 승인하지 않습니다.
4. 후보를 쓰는 명시적 테스트라면 API를 ALLOW_TEST_ANALYSIS=true로 실행해야 합니다. 기존 서버를 중복 실행하지 말고 운영자가 재시작할 때 설정합니다. 아래는 backend에서 실행하는 예시이며 이 작업에서 실행하지 않았습니다.

```powershell
$env:ALLOW_TEST_ANALYSIS = 'true'
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

테스트 종료 후 해당 서버를 중지하고 PowerShell의 `Remove-Item Env:ALLOW_TEST_ANALYSIS`로 세션 설정을 해제한 다음 기본 정책의 서버로 돌아갑니다. 상시 .env를 임의 수정하지 않습니다.

5. 영상 관리 새로고침 후 서버에서 테스트 허용 상태가 확인되면 테스트 모드로 **새 작업을 한 번만** 요청하고 새 job_id를 확인합니다.
6. FastAPI와 별도 터미널(backend)에서 그 새 작업만 Worker로 실행합니다. 아래 placeholder를 실제 새 테스트 job_id로 대체합니다. ROI가 없는 기존 queued ID를 넣거나 --job-id 없이 모든 대기 작업을 실행하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m app.worker `
  --model-path .\out\vision-pipeline-validation-20261112\yolo11m.pt `
  --algorithm-version vision-test-20261112 `
  --allow-test-analysis --once --job-id "<새 테스트 job_id>"
```

algorithm-version은 실행 구분용 레이블입니다. Worker가 실제 AI 소스/모델 해시도 별도 기록합니다. 동시 추론 제한과 작업 선점/복구 정책은 WORKER.md를 따릅니다. 과거 16.16초 영상은 CPU 약17분47초였으며 새59.875초 영상의 처리 시간은 실제 측정 전 확정할 수 없습니다. queued는 추론 시간이 아니라 시작 전 대기 상태입니다.

## 검증

백엔드117개 통과. 프론트엔드에는 초당 평균 소수/0/결측, 탐색 시 버킷 전환, 다른 영상·작업 거부, 미완료 숨김, queued ROI 누락, running/failed 상태, 테스트 비활성 정책, 메인 업로드의 ROI 없는 요청 차단 테스트를 추가했습니다. TypeScript/ESLint/빌드/기존 API 연동을 검사했습니다.

실제 Chrome: 새 밀집 영상에서 queued/ROI 누락/테스트 비활성/요청 비활성/탐지·인원 없음 확인. 기존 완료 영상에서 2초 far2.2/near2.4, 8초 far2.4/near3.9, 12초 far3.8/near2.8이 API와 일치했습니다. 새 영상으로 전환하면 이전 박스와 인원은 즉시 사라집니다. 모바일390px 페이지 넘침 없음, JS 예외0/쓰기 요청0/기존 작업 상태 불변을 확인했습니다.
증거: `C:/Users/Woori/AppData/Local/Temp/crowdpulse-analysis-flow-u8w1s868/` (report.json, new-queued.png, completed-counts.png).

현재 밀집 영상의 새 추론 성공 검증은 안전 조건에 따라 수행하지 않았습니다. 기존 데이터/CSV/작업 상태 변경, 마이그레이션, 모델 다운로드, commit/push/PR도 수행하지 않았습니다.
