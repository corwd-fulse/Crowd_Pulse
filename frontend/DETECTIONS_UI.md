# YOLO 추적 결과 표시

영상 관리에서 영상을 선택하고 "탐지 결과 작업"에서 완료 작업을 선택합니다. 기본값은 해당 영상의 완료 작업 중 첫 항목입니다. 원본 영상/YOLO 탐지/ROI 구역 체크박스를 독립적으로 사용할 수 있습니다. ROI를 숨기면 미리보기 모드로 전환해 재생 조작을 허용합니다. 원본 영상을 숨겨도 외부 재생/일시정지 버튼은 사용할 수 있습니다. 기존 ROI 저장/편집/승인 정책은 유지합니다.

## 데이터 및 동기화

기존 코드에는 박스 표시 기능이 없었습니다. tracker의 실제 CSV 스키마는 `frame,t_ms,track_id,x1,y1,x2,y2,conf`이며 frame은 0부터 시작하고 t_ms는 `int(frame / 원본FPS * 1000)`입니다. FPS는 처리 속도(fps_processed)가 아니라 실행 스냅샷의 원본 FPS를 사용합니다.

새 읽기 전용 API: `GET /api/analysis/jobs/{job_id}/detections`.
응답: analysis_job_id, video_id, fps, frame_count, video_resolution, time_unit, frames 배열(frame,t_ms,boxes). 각 박스는 track_id(없으면 null), x1,y1,x2,y2,confidence를 보존합니다. 서버 경로는 응답에 포함하지 않습니다.

완료 작업과 결과가 있고 worker_token 폴더 및 execution_metadata.json이 DB 실행 스냅샷과 일치하는 CSV만 반환합니다. 다른 작업 파일이나 비슷한 파일명을 임의로 선택하지 않습니다. 누락된 CSV/미완료 작업은 404, CSV 컬럼/좌표/시간 검증 실패는 422로 분리합니다. 헤더만 있는 정상 CSV는 200과 빈 frames입니다. 크기는 64MiB/50만 행/50만 프레임 이하로 제한하고 파일 수정시각·크기별 파싱 결과를 4개까지 캐시합니다. 장시간 대용량 영상은 추후 구간 조회 방식으로 확장할 수 있습니다.

프론트엔드는 requestVideoFrameCallback의 mediaTime으로 재생 프레임을 갱신하고, seeked/구형 브라우저 timeupdate를 보조로 사용합니다. 원본 FPS에 따른 프레임 인덱스와 CSV t_ms를 함께 확인하며 빈 프레임에서 이전 박스를 유지하지 않습니다. requestVideoFrameCallback 미지원 브라우저의 재생 중 갱신 주기는 timeupdate 빈도에 의존합니다. SVG viewBox는 원본 해상도이고 원본 영상과 같은 래퍼를 공유합니다. ID가 없는 탐지는 "ID 없음"으로 표시합니다.

ROI GET/등록 성공 여부는 박스 표시의 조건이 아닙니다. 다만 탐지 자체가 없는 영상에 새 결과를 생성하지 않습니다. 기존 Worker는 ROI 승인 정책을 계속 적용하며 이 UI 때문에 ROI 없는 새 추론 경로를 추가하지 않았습니다. 수동 CSV 가져오기로 완료된 작업처럼 Worker 추적 파일이 없는 작업은 "탐지 CSV 없음"으로 표시됩니다.

## 실제 검증 (2026-11-12)

- 기존 작업 `9bfe14ca-8932-465e-9347-4a50c0fa4904`의 5483행/탐지 있는 402개 프레임을 재사용했습니다. 원본 총 프레임은 404이며, 탐지 없는 프레임은 박스를 표시하지 않습니다.
- 25FPS, 1920×1080에서 2초(프레임50,12개), 8초(프레임200,11개), 12초(프레임300,20개)의 SVG 좌표·ID가 API/CSV와 일치했습니다. 8초 스크린샷에서 사람 위치와 박스 정렬을 확인했습니다. 이는 전체 검출 정확도/ID 스위칭 품질 평가가 아닙니다.
- 영상 재생 중 박스 변경, 표시 토글 세 가지, 모바일390px 페이지 넘침 없음, JS 예외 없음, 쓰기 요청 없음 확인.
- 증거: `C:/Users/Woori/AppData/Local/Temp/crowdpulse-detections-ui-69wq6cvb/` (report.json, frame-2.png, frame-8.png, frame-12.png, mobile.png).
- 실제 서버와 Sandbox Worker 실행 계정이 달라 tempfile 실행 폴더의 소유자 전용 ACL이 조회를 막았습니다. 기존 검증 작업 폴더 하나에만 `DESKTOP-7GB9D1P\Woori` 읽기·탐색 권한(RX)을 부여했습니다. CSV/DB 내용은 변경하지 않았습니다. 다른 계정으로 API/Worker를 운영할 경우 별도의 최소 읽기 권한 구성이 필요합니다. API는 권한을 자동 변경하지 않습니다.

검증: pytest, npm test, npm run test:live, tsc -b, ESLint, Vite build. 실제 추론/모델 다운로드/마이그레이션은 수행하지 않았습니다.

## 메인 대시보드 플레이어 연결 (2026-11-12)

이전 구현은 영상 관리의 VideoRoiEditor에만 DetectionOverlay를 붙였습니다. 메인 대시보드는 별도 CameraViewer가 URL.createObjectURL로 만든 로컬 blob 영상을 계속 재생했고, trackedJob은 업로드 요청 상태 갱신에만 사용했습니다. 선택한 DB 분석 작업과 플레이어 소스가 연결되지 않아 탐지 API 호출/토글/박스가 없었습니다.

이제 CameraViewer는 선택 작업의 video_id에 대한 서버 원본 스트림을 MonitoringPlayer로 재생합니다. MonitoringPlayer는 기존 DetectionOverlay를 그대로 재사용하고, 해당 작업 결과에 저장된 ROI 스냅샷만 사용합니다. 현재 영상의 변경 가능한 ROI 대신 분석 당시 스냅샷을 표시하여 이력을 혼용하지 않습니다. 작업 변경 시 플레이어를 job.id로 재마운트하여 이전 프레임/ROI/CSV를 버립니다. 응답 job_id/video_id/해상도가 다르면 오버레이를 표시하지 않습니다. 미완료 작업에는 탐지 API를 요청하지 않습니다.

로컬 업로드 미리보기는 별도 경로로 유지합니다. 파일 선택 즉시 로컬 미리보기/업로드 중으로 표시하고, 업로드 성공과 분석 완료를 구분합니다. 로컬 파일에는 현재 선택한 다른 작업의 탐지를 절대 겹치지 않습니다. "선택 작업의 서버 영상 보기"/"업로드 영상 미리보기"로 소스를 명시적으로 바꿀 수 있습니다. 다른 작업을 선택하면 선택 작업의 서버 영상으로 전환합니다.

서버 플레이어에는 원본/YOLO/ROI 토글, 재생/일시정지, 작업/영상 식별자, 테스트/ROI 승인/방향 품질 표시를 추가했습니다. 원본 비율 래퍼와 공통 SVG viewBox를 사용합니다. 기존 대시보드 .monitor-stage의 240px 고정 높이는 이 플레이어에서 자동 높이로 덮어써 문구/영상 겹침을 방지합니다.

실제 메인 대시보드 URL: `http://localhost:5173/dashboard?job=9bfe14ca-8932-465e-9347-4a50c0fa4904`.
Chrome 1440×1000 및 390×844에서 8초/프레임200/박스11개/ID를 확인했습니다. 2초50프레임12개·8초200프레임11개·12초300프레임20개 각각의 좌표와 ID가 기존 CSV 응답과 일치합니다. 재생 중 갱신과 원본/YOLO/ROI 토글도 확인했습니다. 다른 영상의 queued 작업 선택 시 소스가 변경되고 박스가 0개가 됐습니다. API 탐지 응답은 200, 브라우저 JavaScript 예외/쓰기 요청은 0건입니다. 테스트에서는 로컬 업로드를 Mock 처리해 소스 분리도 검증했습니다. DB/CSV 변경 및 추론은 수행하지 않았습니다.

증거: `C:/Users/Woori/AppData/Local/Temp/crowdpulse-dashboard-boxes-3mhn46m3/`.
report.json은 프레임/좌표/ID 비교 결과이며, 고정 높이 수정 후 최종 레이아웃/ROI 토글 결과는 layout-final.json입니다. 최종 스크린샷은 dashboard-8s.png와 dashboard-mobile.png입니다. 기존 report.json의 roiHidden=null은 자동화 스크립트의 잘못된 selector 표현식 때문이며, layout-final.json에서 올바른 선택자로 ROI 숨김을 확인했습니다.

## 작업 ID 없는 대시보드의 기본 선택 (2026-11-12)

기존 DashBoard는 job 쿼리가 없으면 작업 목록의 jobs[0]을 선택했습니다. 기존 목록 API는 생성 시각 내림차순/최대100개여서 가장 최근 생성된 queued 작업이 선택됐습니다. 당시 실제 목록의 앞 두 작업도 queued였습니다. MonitoringPlayer가 미완료 작업에 박스를 표시하지 않는 동작 자체는 정상입니다.

기존 목록 API에 선택적 `status`, `order_by=created_at|finished_at`, `limit=1..100`, `offset` 조회 조건을 추가했습니다. 기본 요청의 계약은 그대로입니다. `GET /api/analysis/jobs?status=succeeded&order_by=finished_at&limit=1`은 최근100개 필터링 이전 전체 작업에서 완료 시각 내림차순으로 한 건을 찾습니다. 따라서 오래 전에 생성됐으나 최근에 완료된 작업도 누락하지 않습니다.

`/dashboard`는 해당 최신 완료 작업을 기본 선택하고 기존 다른 쿼리를 보존하며 job을 URL에 replace로 기록합니다. URL에 job이 있으면 상태와 관계없이 그 작업을 우선하며 목록 밖 ID는 기존 상세 API로 조회합니다. 직접 고른 queued/running 작업을 최신 완료 작업으로 되돌리지 않습니다. 완료 작업이 없으면 자동 선택하지 않고 "분석 완료된 영상이 없습니다"를 표시합니다. 목록 새로고침과 페이지 새로고침은 명시된 URL 작업을 유지합니다. 어떤 조회도 작업 상태를 변경하지 않습니다.

서버 플레이어에서 해당 video_id의 영상 상세 GET을 사용해 실제 파일명도 표시합니다. 이름 응답의 영상 ID가 다르면 사용하지 않습니다. 기존 작업별 박스/ROI/인원 초기화와 응답 ID 검증은 그대로 사용합니다.

Chrome 검증: `/dashboard`와 명시적 `?job=9bfe14ca-8932-465e-9347-4a50c0fa4904` 양쪽에서 동일 영상/작업이 연결되고 8초/프레임200/박스11개/ROI2개/평균 인원 지표가 확인됐습니다. queued 작업 선택 및 실제 새로고침 후 박스0/ROI0과 선택 유지, 완료 작업 새로고침 후 박스 복원, 모바일390px 가로 넘침 없음을 확인했습니다. 기존 실제 작업 목록/상태는 전후 비교에서 동일했고 POST/PUT/DELETE는 없었습니다.

증거: `C:/Users/Woori/AppData/Local/Temp/crowdpulse-default-job-m7_zb7pz/` (default.png, explicit.png, mobile.png, jobs-before.json, report.json, reload-final.json). 첫 자동화 실행의 reload snapshot은 문서 교체 중 읽혀 null이었으므로 새 performance.timeOrigin과 영상 readyState까지 대기한 reload-final.json을 최종 새로고침 검증 근거로 사용합니다. 데이터 변경이나 추론은 수행하지 않았습니다.
