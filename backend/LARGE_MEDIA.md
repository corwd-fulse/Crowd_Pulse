# 장시간 영상·대용량 분석 데이터 운영

이번 변경은 분석 데이터 입출력과 재생 경로에만 적용한다. YOLO/ByteTrack/Optical Flow 및 Forecast 모델, 기존 가상환경, DB migration은 변경하지 않는다. 원본 영상·분석 CSV를 재생용으로 덮어쓰지 않는다.

## 리플레이

`GET /api/analysis/jobs/{job_id}/replay?start_ms=1800000&end_ms=1805000`

`GET /api/analysis/jobs/{job_id}/detections?start_ms=1800000&end_ms=1805000`

- 기본 재생 구간 5초, 요청 가능 최대 30초. 서버가 반환하는 `window.end_ms`와 `next_start_ms`가 실제 구간 경계다. 고밀도 구간은 완전한 프레임 경계에서 더 짧게 반환한다. 임의의 사람/행을 샘플링하지 않는다.
- 프레임 방향 표시를 위해 이전 1프레임도 포함한다. `frame_count`와 `fps`는 전체 영상 기준이다. `frames` 배열 위치를 전체 프레임 번호로 해석하면 안 된다.
- 최초 요청은 검증된 디스크 인덱스를 비동기로 생성하면서 HTTP 202와 `Retry-After`를 반환할 수 있다. 클라이언트는 최대 5분까지 재조회하며, 실패 시 오류를 표시한다. 대형 자료는 최초 인덱스 생성 시간이 필요하다.
- 원본 tracker CSV, 프레임 인덱스, 실행 스냅샷, ROI 및 fingerprint에 묶인 SQLite 파생 인덱스다. 저장 위치는 `backend/uploads/replay-index/`. 원본 파일이 바뀌면 새 키를 사용한다. 기록된 transition 이벤트가 있으면 같은 정책·이벤트인지 대조한다.
- 인덱싱은 프로세스당 1개 실행, 최대 4개 대기/실행 항목. SQLite 캐시 16MiB, CSV 삽입 2,000행 배치. OS 파일 잠금은 프로세스 종료 시 해제된다. `.lock` 파일이 남는 것 자체는 실행 중이라는 뜻이 아니다.
- 원본 추적 CSV 8GiB/1억 행, 프레임 수 1천만, 파생 인덱스 24GiB, 단일 프레임 2만 관측의 명시적 자원 예산을 둔다. 응답은 최대 5만 포인트/12MiB, 이벤트 이력·경로 ID 목록은 각각 2만 건이다. 초과 시 원본을 보존하고 오류를 반환한다.
- 작은 기존 무구간 요청은 이전 응답을 유지한다. 8MiB 초과 추적 CSV 또는 50만 초과 프레임의 무구간 요청은 첫 5초 구간으로 전환한다. 새 클라이언트는 항상 구간을 명시한다.
- 프런트는 다음 구간을 약 2초 전에 요청하고, 이전 구간을 정리한다. 공유 캐시는 12개/문자열 예상 크기 24MiB로 제한한다. 컴포넌트·렌더러·네트워크·JS 객체까지 포함한 브라우저 전체 RSS의 상한이라는 뜻은 아니다.
- 관제탑의 최근 10/30/60초 이벤트 집계를 위해 이벤트 이력은 구간 시작 이전 60초를 포함한다. 누적 유입/유출은 디스크의 이전 이벤트를 합산하므로 페이지마다 초기화하지 않는다. 전체 경로 필터를 위한 현재 ID별 경로 소속도 별도로 제공한다.
- 조건부 집중 추정의 대형 입력은 `indexed_context.py`에서 프레임을 지연 조회한다. 기존 gate 판정과 결측 정책을 유지하며, 이전 프레임에 없었던 ID의 상태는 정리한다. 전체 추적 포인트를 메모리에 모으지 않는다. gate 이벤트는 별도 10만 건 예산을 유지한다. 인덱스 준비 전에는 관측 불가로 표시하며 추정값을 만들어내지 않는다.

## CSV와 학습용 추출

- 결과 CSV import는 파일당 2GiB, 전체 4GiB, 파일당 1억 행까지의 제한된 스트리밍 경로다. 기존 64MiB 상수는 레거시 참조용이며 현재 importer는 `MAX_STREAM_FILE_BYTES`를 사용한다.
- 키 중복 검사는 Python 전체 `set` 대신 임시 SQLite에서 수행한다. 소형 metric은 기존 처리, 1만 행 초과 metric은 반복 가능한 `DiskRows`, DB 저장은 1,000행 배치 upsert다. metric 종류별 컬럼만 갱신하여 NULL/기존 다른 측정값을 덮어쓰지 않는다. 원본 개인 방향 행은 DB에 대량 복제하지 않고 검증된 CSV로 보존한다.
- 8MiB 초과 tracker CSV는 `streaming_tracks.process()`로 외부 정렬한다. ID/프레임 순서와 앞뒤 15개 관측을 유지하고, 기존 5프레임 방향·ID switch·체류/변위 판정 수식을 재사용한다. 청크마다 ID를 새로 시작하지 않는다. 같은 ID/프레임 중복은 원본을 보존한 채 오류로 처리한다.
- 개인 방향 CSV는 순차 기록한다. 원형 평균은 SQL 집계, ID/초별 투표는 디스크 중간 테이블로 처리한다. 한 track-second 10만 관측, zone-second 2만 투표, 전체 100만 zone-second 메타데이터 예산을 둔다. 이는 원본 사람 행을 샘플링하는 기능이 아니다.
- 원래 5개 CSV 헤더와 count/inflow/outflow/Hybrid 산식은 유지한다. Optical Flow 구현·정책은 변경하지 않는다. FULL의 디코딩·Optical Flow 및 결과 메타데이터 전체가 일정 메모리를 사용한다는 보장은 하지 않는다.
- `GET /api/capture/venues/{venue_id}/sessions/{session_id}/dataset.csv?interval=5`를 그대로 사용한다. interval은 1/5/10/30/60초, camera_id/zone_id 필터를 지원한다.
- DB cursor는 1,000행 단위, 집계는 완료된 과거 버킷 단위로 처리한다. 출력은 임시 디스크 CSV + FileResponse로 전달하고 새 임시 파일만 응답 후 정리한다. 응답 전체를 StringIO에 보관하지 않는다. 25만 출력 행/256MiB 제한을 유지한다.
- 결측은 빈 셀, 저신뢰 Flow는 null, 세션/카메라 간 합산과 rolling 상태 공유는 하지 않는다. 기존 future leakage 우려가 있는 전체 영상 정규화 bottleneck score와 미보정 방향 count는 계속 학습 특징에서 제외한다. 원시 offline 방향 판정의 앞뒤 관측과 실시간 인과적 Forecast 입력은 구분한다.

## 코덱과 재생용 복사본

`GET /api/videos/{video_id}/media-info`: ffprobe 실제 코덱/픽셀 형식/FPS, 재생용 사본 상태.

`POST /api/videos/{video_id}/playback`: 원본을 보존하고 H.264/yuv420p MP4 사본 생성 요청(202).

`GET /api/videos/{video_id}/playback/stream`: 전체 프레임 검증에 성공한 사본만 제공. Range 요청 지원.

- 기존 `/stream`은 계속 원본이다. Worker의 `storage_path`와 분석 대상은 변경하지 않는다.
- 서버 플레이어와 촬영 세션 플레이어는 브라우저 오류가 발생하면 코덱/변환 안내를 표시한다. 검증된 사본이 이미 있으면 재사용하고, 없으면 운영자가 변환 버튼을 누른다. 촬영 세션 사본 로드 시 공통 시간축의 해당 오프셋으로 이동한다.
- FFmpeg와 ffprobe는 `CROWD_FFMPEG_BINARY`, `CROWD_FFPROBE_BINARY`로 지정할 수 있다. PATH 또는 프로젝트 `.tools/ffmpeg/*/bin/*.exe`도 탐색한다. Python 패키지나 기존 가상환경을 변경하지 않는다.
- 이 워크스페이스에는 FFmpeg 공식 다운로드 페이지에서 연결한 Gyan essentials 9.0.2 portable build를 `.tools/ffmpeg/ffmpeg-9.0.2-essentials_build/bin/`에 배치했다. ZIP SHA-256은 `60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba`, 제공된 checksum과 일치한다. `.tools/`는 Git 제외다. 다른 서버에서는 실행 파일을 별도 준비해야 한다.
- CPU 변환, 프로세스당 동시 1개, 인코더 2스레드, 1시간 timeout, 입력 8GiB/출력 16GiB 예산 및 디스크 여유 검사를 적용한다. 여러 API 프로세스는 다른 영상을 동시에 변환할 수 있다. Vision Worker/GPU를 사용하거나 종료하지 않는다.
- 변환 전후 SHA-256, 해상도, 디코딩된 프레임 수와 **모든 프레임의 상대 PTS**를 검증한다. 허용 오차 20µs, 재생 사본 첫 프레임 시간은 0이어야 한다. 사본은 검증 후 공개한다.
- 가변 프레임 간격(VFR), 프레임 수/시간 불일치, yuv420p에서 유지할 수 없는 홀수 해상도는 명시적으로 실패한다. VFR을 조용히 CFR로 바꾸어 프레임을 추가/삭제하지 않는다. 실제 CCTV에서 해당 형식이 확인되면 원본 PTS 기반 리플레이를 별도로 검토해야 한다.

FFmpeg 출처: https://ffmpeg.org/download.html / https://www.gyan.dev/ffmpeg/builds/

## 실제 영상 도착 후

1. 업로드 전 원본 크기/코덱/FPS/촬영 시각·카메라 오프셋과 여유 공간을 확인한다. 대형 업로드는 기존 촬영 세션 업로드(2GiB)를 사용한다. 레거시 multipart 업로드는 기존 200MiB 제한을 유지한다.
2. ROI와 촬영 세션을 확인하고 기존 분석 큐에 등록한다. 기존 GPU Worker 병렬 정책을 임의로 늘리지 않는다.
3. 30분~1시간 실제 밀집 영상으로 YOLO/ByteTrack/FULL Optical Flow, VRAM/RSS, 장시간 처리 속도, 실패/재시도를 측정한다. 합성 I/O 테스트 시간을 AI 추론 시간으로 해석하지 않는다.
4. 최초 인덱스 준비 후 시작/중간/끝 및 페이지 경계에서 원본·2D·3D를 비교한다. 필요 시 재생용 변환을 요청한다.
5. CSV와 스키마를 내보내고 결측/ID 단절/구역 보정/훈련-검증 세션 분리를 Forecast 담당자와 확인한다. 합성 자료는 학습에 넣지 않는다.

검증 증거와 성능 측정은 `validation/large_media_20261117/REPORT.md`에 기록한다.
