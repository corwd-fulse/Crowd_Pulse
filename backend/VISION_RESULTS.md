# Vision CSV 결과 저장

실제 큐 실행 연결과 운영/테스트 ROI 정책은 [WORKER.md](WORKER.md)를 참고한다.
Worker는 별도 CLI이며, 이 CSV 가져오기 API 자체는 추론을 실행하지 않는다.

## CSV 스키마와 알고리즘 버전 분리

추가 리비전 `20261112_03`은 병목 윈도우의 10초 배수 제약만 완화한다. 기존 행은 변경하지 않는다.
저장 계층은 출력 스키마만 검증하며 AI 함수나 임계값을 가져오지 않는다. `window_start`는
영상 내 비음수 초이며, 10초 등 윈도우 길이는 알고리즘 실행 설정의 책임이다.

새 import의 `execution_metadata.algorithm_version`은 필수다(팀 릴리스/커밋 식별자 권장).
`model_version`과 구분한다. `csv_schema_version`은 현재 `"1"`이며,
`settings`에 집계 방식과 모듈별 버전·설정을 추가할 수 있다. 방향 집계 방식이 바뀌어도
CSV 스키마가 같으면 DB 변경 없이 새 analysis_job으로 저장한다. 과거 결과는 덮어쓰지 않는다.

각도는 0 ≤ angle < 360, 방향/변위는 유한 실수, 시간·유입·유출·표본 수는 정수,
평균 인원은 소수로 검증한다. 360도는 0도와 동치이므로 스키마의 표준 표현은 0도다.
`low_confidence`는 DB bool로, 원시 `reliable`은 관리 CSV 바이트로 보존한다.
파서·SQL 타입·정밀도·저장/조회 왕복은 테스트로 검증한다.

방향 품질은 ROI 승인·분석 성공 여부와 별개다. 실행 메타데이터의 `direction_quality`는
`unvalidated`(기본값), `validated`, `rejected`다. `validated`에는
`direction_review.reviewer`와 `direction_review.evidence`가 필요하다.
API의 `direction_presentation`과 행별 `flow_direction_confirmed`/
`individual_direction_confirmed`를 사용하고, 숫자가 존재하거나 job이 succeeded라는
이유만으로 확정된 방향으로 표시하지 않는다. 승인된 ROI와 검증 근거가 모두 있어야
표시 확인 상태가 가능하며, 낮은 신뢰도 Flow와 없는 방향은 확인 상태로 표시하지 않는다.
이는 메타데이터에 기록된 검토 상태이며 서버가 정확도를 자동 입증하는 것은 아니다.
과거 스냅샷에 품질 필드가 없으면 미검증, 알고리즘 버전이 없으면 legacy-unspecified로 조회한다.

아래 초기 설계 설명의 '10초 버킷'은 당시 파이프라인 설정 설명이며 현재 DB 제약은 아니다.

이 단계는 완료된 CSV를 명시적으로 가져오는 기능이다. YOLO 실행, worker,
자동 스케줄링은 하지 않는다. 기존 작업 생성·조회 API는 그대로 유지한다.

## 데이터 구조

| 테이블 | 종류 | 내용 |
|---|---|---|
| vision_results | 일반 PostgreSQL | 작업당 1건, ROI·실행 스냅샷, 실제 분석 시각, 수신 시각, CSV manifest, fingerprint |
| vision_zone_metrics | TimescaleDB Hypertable | 시간·구역별 평균 인원/유입/유출/Optical Flow/개인 방향 집계 |
| vision_bottleneck_scores | TimescaleDB Hypertable | 10초 버킷별 병목 점수·구성 요소·위험 등급 |

`vision_results.analysis_job_id`가 PK이자 기존 `analysis_jobs`의 FK다.
영상은 `analysis_jobs.video_id → videos.id`로 연결하여 영상 ID 불일치를 방지한다.
같은 영상의 재분석은 새 작업으로 구분한다. 영상별 ROI 이력은 작업과 결과를 조인해 확인한다.

두 Hypertable은 NOT NULL `imported_at`으로 분할한다. PK는
`(analysis_job_id, imported_at, region_key, video_offset_ms/window_start_ms)`이며
모든 고유 제약에 분할 시각을 포함한다. FK는 일반 테이블만 참조하고 삭제는 RESTRICT다.
조회는 결과 헤더의 `imported_at`도 조건에 넣어 청크를 제한한다.

`imported_at`은 실제 DB 수신 트랜잭션 시각이다. 실제 분석 시각은 별도의
`analysis_started_at`, `analysis_finished_at`이며 호출자가 시간대가 포함된 값으로
명시해야 한다. 과거 실행의 시각을 모르면 업로드 시각·수신 시각·소요 시간으로 만들지 않는다.
촬영 시각은 기존 `videos.recorded_at`에 남으며 이 importer는 생성하거나 변경하지 않는다.

시계열의 `video_offset_ms`는 CSV `sec * 1000`, `window_start_ms`는
`window_start * 1000`이다. 초 단위 CSV 원본은 파일로 함께 보관한다.

## 의미와 결측값

- `mean_person_count`: 초당 평균 인원. `NUMERIC(14,4)`로 저장하고 지원 정밀도 초과는
  조용히 반올림하지 않고 거부한다. 기존 정수형 `crowd_metrics.person_count`와 구분한다.
- `inflow/outflow`: 유입·유출 수치. 비음수 정수.
- `flow_angle_deg/magnitude_px/motion/low_confidence`: Optical Flow 출력 그대로 저장.
  전부 결측인 flow는 `low_confidence=True`일 때 허용한다.
- `mean_angle_deg/direction_sample_count`: CSV `mean_angle_deg/n_tracks`.
  표본 수는 고유 인원수가 아니다. 부재한 구역/초는 0으로 만들지 않는다.
- `has_observation/has_flow/has_individual_aggregate`는 해당 CSV에 행이 있었는지 표시한다.
  세 CSV의 키를 전체 합집합으로 결합하므로 없는 행과 실제 0을 구분한다.
- `score`는 병목 연구 지표이며 `congestion_percent`가 아니다. 기존 혼잡도 컬럼에 넣지 않는다.
- 위험 등급은 원문과 매핑을 함께 보존한다: `낮음→low`, `주의→medium`, `높음→high`.
  `critical`은 이 파이프라인이 출력하지 않는다. 점수 행을 병목 이벤트로 자동 변환하지 않는다.
- 점수 CSV가 헤더만 있는 경우에도 정상 완료다. 원시 개인 방향·개인 집계도 빈 파일을 허용한다.
  구역 관측 CSV는 구성된 모든 구역을 최소 한 번 포함해야 한다.
- `reliable=False` 개인 방향의 각도·변위 결측은 정상이다. `reliable=True`인 정지 표본도
  `speed_px=0, angle_deg=NULL`을 허용한다. NaN/Infinity 숫자는 허용하지 않는다.
- `speed_px`는 관측 창의 픽셀 변위이며 m/s나 px/s로 해석하지 않는다.

기존 `crowd_metrics`, `crowd_predictions`, `bottleneck_events`는 변경하거나 채우지 않는다.
평균 관측·병목 점수와 순간 인원·미래 예측·병목 이벤트의 의미가 다르기 때문이다.

## 개인별 원시 방향 보관 방식

전체 DB 행 적재 대신 **관리되는 CSV 사본 + manifest**를 선택했다. 현재 검증 영상만 해도
원시 방향은 5,280행이고 구역 집계는 33행이다. 장시간 영상·여러 영상에서 원시 궤적은 크게
증가한다. 현재 제품의 조회 대상은 구역 시계열이므로 DB에는 집계를 저장하고, 재검증·추후
재집계에 필요한 원시 행은 원본 CSV로 보존한다.

모든 CSV의 상대 경로, SHA-256, 바이트 크기, 행 수, 컬럼별 결측 수를 JSONB manifest에 저장한다.
원시 `reliable` 값과 NULL 표현은 파일 바이트 그대로 보존되며 다운로드 API로 받을 수 있다.
추후 track ID별 대화형 검색이 필요해지면 별도 보존 기간·전용 테이블을 설계한다.
원시 방향의 키 중복 검사는 스트리밍하며 키 집합만 유지하고 전체 행은 메모리에 적재하지 않는다.

## API

| 메서드 | 경로 | 동작 |
|---|---|---|
| POST | /api/analysis/jobs/{job_id}/results | CSV 5개 및 스냅샷 명시적 저장 |
| GET | /api/analysis/jobs/{job_id}/results | 영상 연결, ROI·실행 시각·단위·manifest |
| GET | /api/analysis/jobs/{job_id}/results/zone-metrics | 구역 시계열 조회 |
| GET | /api/analysis/jobs/{job_id}/results/bottleneck-scores | 점수 조회; 빈 결과는 items=[] |
| GET | /api/analysis/jobs/{job_id}/results/artifacts/{kind} | 관리되는 CSV 사본 다운로드 |

POST는 `multipart/form-data`다. 파일 필드는 확장자를 제외한 아래 이름을 정확히 사용한다.

```text
crowd_observations, flow, individual_directions, individual_zone_agg, bottleneck_scores
```

텍스트 form 필드:

- `roi_snapshot`: JSON 문자열. 기존 Zone `name/polygon` 구조와 `video_resolution`,
  `status`, `approved_for_analysis`를 포함한다. 후보는 `candidate` 또는 `candidate_unvalidated`
  + false, 승인은 `approved` + true로 명시한다. importer가 후보를 승인하지 않는다.
  원본 JSON 파일을 수정하지 않고 받은 스냅샷을 저장한다. 기존 ROI의 이미지 경계 꼭짓점
  `(x=width 또는 y=height)`도 폴리곤 경계로 허용한다.
- `execution_metadata`: 아래 필드의 JSON 문자열.

| 필드 | 의미 |
|---|---|
| analysis_started_at / analysis_finished_at | 기록된 실제 분석 시작·종료, 시간대 필수 |
| model_version | 실행에 사용한 모델 버전 문자열 |
| video_sha256 | 등록된 원본 파일의 SHA-256, 소문자 64자리 |
| fps / frame_count | 원본 FPS·총 프레임 수; 상대 시간 검증에 사용 |
| settings | 추론 크기·tracker 설정·device·모델 해시 등을 담는 실행 설정 객체 |

ROI에 source_sha256/source_fps/source_frame_count가 있으면 실행 메타데이터와 비교한다.
등록된 영상 파일의 실제 SHA-256도 비교한다. importer는 영상 추론을 실행하지 않으며
CSV 수치가 실제 영상에서 계산됐다는 사실이나 분석 시각의 진위까지 입증하지는 않는다.
호출자는 실제 실행에서 수집한 정보를 전달해야 한다. 수기 추정 시각은 사용하지 않는다.

파일당 64MiB, 합계 256MiB, JSON 문자열 각각 100만 문자 제한이다. 원시 개인 방향은 최대
100만 행, 나머지는 각각 최대 10만 행이다. 파일명/서버 경로를 클라이언트 입력으로 사용하지 않는다.
컬럼 순서, 유한 숫자·범위·정수, 엄격한 bool, 구역명, CSV 키 중복, 영상 시간 범위를 검증한다.

새 저장은 201, 동일 작업·동일 바이트와 스냅샷의 재요청은 200(`created=false`),
다른 결과 덮어쓰기는 409다. 누락 작업/결과는 404, 잘못된 입력은 422, 크기 초과는 413,
DB/파일 저장 실패는 내부 정보를 숨긴 503이다.

저장 성공 시 해당 작업을 `succeeded`로 바꾸고 전달받은 실제 시작·종료 시각을 사용한다.
기존 시각이 다르면 덮어쓰지 않고 409다. failed/cancelled 작업에 새 결과는 넣지 않는다.
점수 행이 없어도 정상 완료할 수 있다. ROI가 candidate인 성공 결과는 품질 승인과 다르다.

시계열 조회: `region_key`, `start_ms`, `end_ms`(양끝 포함), `limit`(1–1000), `offset`.
점수 조회: `region_key`, `limit`, `offset`. 결과는 시간·구역 순으로 정렬한다.
HTTP 응답에 서버 파일 경로를 노출하지 않고 artifact 다운로드 URL을 제공한다.

## 트랜잭션·보존

작업 행을 FOR UPDATE로 잠그고 작업당 결과 PK와 fingerprint로 중복 저장을 막는다.
CSV 5개 전체 검증 후 `uploads/results/{job_id}/import-.../`에 사본을 저장하고,
복사본 해시를 다시 확인한 뒤 헤더·시계열·작업 상태를 한 트랜잭션으로 commit한다.
일반적인 실패에는 전체 DB rollback과 이번 요청의 새 사본 디렉터리 정리를 수행한다.
기존 결과와 파일은 삭제하거나 교체하지 않는다.

파일시스템과 DB는 원자적 트랜잭션이 아니므로 프로세스 강제 종료 시 고아 디렉터리가 남을 수 있다.
자동 삭제·정리 작업은 추가하지 않았다. CSV 파일도 DB와 함께 백업해야 한다.
결과가 성공한 뒤 외부에서 CSV를 수정/삭제하지 않도록 관리해야 한다. manifest 해시로 검증 가능하다.
다른 작업의 완전한 중복 결과를 deduplicate하지는 않는다(재분석 이력 구분).

## 마이그레이션 및 검증

신규 리비전 `20261112_02`는 `bdebb741521f` 다음이다. 기존 리비전은 변경하지 않는다.
확장을 설치하거나 서버 설정을 변경하지 않으며 새 테이블만 생성한다. 데이터 보존을 위해
파괴적인 downgrade는 차단한다. 앱 시작 시 schema 생성이나 자동 import는 하지 않는다.

```powershell
# backend/에서 가상환경 Python으로 실행
python -m alembic upgrade head
python -m alembic check
python -m pytest -q
```

테스트는 기존 PostgreSQL/TimescaleDB에서 외부 rollback 트랜잭션과 임시 파일 디렉터리를
사용한다. 테스트 시작·종료 시각은 fixture용이고 운영 분석 이력으로 기록되지 않는다.
저장·조회·CSV 다운로드, 저장소 sample CSV, 정상 NULL·빈 점수, 중복·충돌, 잘못된 CSV,
영상 해시 불일치, 삽입/commit 실패 rollback, 기존 health·영상·작업 API를 검증한다.
YOLO 추론은 테스트하지 않는다.
