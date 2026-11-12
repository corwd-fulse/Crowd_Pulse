# 실제 AI 결과 대시보드

대시보드는 `/api/dashboard/summary`의 mock 값을 실제 결과로 사용하지 않는다.
작업을 선택하면 succeeded 상태에서만 다음 API를 조회한다.

```text
GET /api/analysis/jobs/{job_id}/results
GET /api/analysis/jobs/{job_id}/results/zone-metrics?limit=1000&offset=...
GET /api/analysis/jobs/{job_id}/results/bottleneck-scores?limit=1000&offset=...
```

응답 스키마는 현재 백엔드 코드와 저장된 Worker 테스트 결과를 직접 확인해 정의했다.
지표·점수는 페이지를 모두 읽고, 다른 job_id가 반환되면 거부한다. 결과 조회 취소와 작업 ID 검사를
통해 선택을 빠르게 바꿔도 이전 결과가 섞이지 않는다. 로딩·조회 실패·빈 결과를 구분하며
실패/취소 작업에는 결과를 요청하지 않는다. API 오류 시 Mock으로 대체하지 않는다.

## 화면

- 대시보드 작업 선택: 최신 100개 작업. `/dashboard?job={id}`로 목록 밖의 과거 작업도 조회한다.
- 영상 관리: 기존 5초 폴링을 공통 훅으로 재사용하며 완료 작업의 대시보드 링크와 실패 메시지를 제공한다.
- 업로드/분석 요청: 기존 기능 유지. 대시보드에서 새 요청을 만들면 그 작업을 자동 선택한다.
- 그래프: 구역별 초당 평균 인원을 소수로 표시한다. X축은 영상 상대 시간이며 촬영 시각이 아니다.
  NULL 관측은 0으로 바꾸거나 연결하지 않는다. 실제 예측값을 생성하지 않는다.
- 요약: 구역·초별 평균의 최댓값, 유입/유출 합계, 최고 병목 점수. 현재 순간 인원·고유 인원·혼잡도
  퍼센트로 해석하지 않도록 단위를 표시한다.
- 지표 표: 구역 필터, 20행씩 페이지 이동, 유입·유출·방향 품질·방향 표본 수.
- 병목: 점수·한국어 위험 등급·구성 요소. 빈 점수는 정상 완료 가능 상태로 표시한다.
- test, 미승인 ROI, 미검증 방향 상태를 명확히 표시한다.
- 미검증·사용 제한 방향은 숫자 화살표로 표현하지 않는다. low_confidence=true는 신뢰 낮음,
  NULL/없는 버킷은 데이터 없음이다. global/row 확인 상태, 승인 ROI, validated 품질을 모두 만족하는
  값만 검증된 방향 **추정값**으로 표시한다.
- 개인 방향 reliable과 원시 NULL은 CSV 다운로드로 보존한다. n_tracks는 표본 수,
  speed_px는 관측 창의 픽셀 변위이며 실제 속도가 아님을 표시한다.
- 기존 지도·위험·예측 Mock은 사용자가 별도 시연 영역을 열 때만 렌더링하며 실제 결과와 분리한다.

## 실행·검증

```powershell
# frontend/
npm.cmd run dev
npx.cmd tsc -b
npm.cmd run lint
npm.cmd run build
npm.cmd test
npm.cmd run test:live
```

기본 API는 `http://127.0.0.1:8000`이며 `VITE_API_BASE_URL`로 바꿀 수 있다.
백엔드의 현재 CORS 설정에 맞게 개발 화면은 `http://localhost:5173`에서 연다.

저장된 실제 테스트 결과 바로가기:

```text
http://localhost:5173/dashboard?job=9bfe14ca-8932-465e-9347-4a50c0fa4904
```

`test:live`는 위 보존된 작업을 GET으로만 읽어 34개 지표·4개 점수·5개 artifact와 정책 상태를
확인한다. DB 쓰기·Worker/추론 실행은 하지 않는다. 기본 `npm test`에서는 이 live 테스트를 건너뛴다.
Live UI 테스트는 API Mock 없이 React 대시보드를 실제 GET 응답으로 렌더링하며, 그래프 두 선·정책 배지·
Mock 부재를 검증한다. jsdom의 레이아웃 측정만 고정 크기로 대체한다.
테스트 fixture JSON은 위 실제 응답에서 가져온 테스트 전용 사본이며 제품 화면에는 import하지 않는다.

React 테스트는 완료 전 조회 방지, 5초 폴링 완료 전환/종료/숨김 탭, 테스트·미승인·미검증 표시,
NULL·낮은 신뢰도, 소수·0 보존, 에러 재조회, 응답 순서 역전, 페이지 조회, Mock 분리와 실제 Recharts
SVG 선 생성을 검증한다. 레이아웃 측정만 고정 크기로 대체한다.

현재 단계에서는 ROI 등록·승인 UI, 테스트 모드 작업 생성 UI를 추가하지 않았다.
새 업로드의 운영 분석에는 별도로 승인된 영상 ROI 등록과 Worker 실행이 필요하다.
이번 변경은 백엔드·DB·알고리즘·기존 결과 데이터를 변경하지 않는다.
