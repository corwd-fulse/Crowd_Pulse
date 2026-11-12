# 샘플 출력 CSV

`app/services/pipeline.py`의 `run_zone_pipeline()`이 만드는 5개 결과 파일의 실제 출력
샘플이다. 테스트 데이터는 Drive의 `E05_001.avi`(할로윈 인파밀집, 같은 카메라로 `E05_003`도
있음) + `configs/zones/alley_entrance_inner.json`의 구역 정의로 만들었다.
전부 SAMPLE/MOCK이 아니라 실영상 분석 결과지만, 데모용 테스트 데이터이지 운영 데이터는
아니다.

| 파일 | 생성 모듈 | 비고 |
|---|---|---|
| `crowd_observations.csv` | `zone_analysis.py` | 전체(61초) |
| `flow.csv` | `optical_flow.py` | 전체 |
| `individual_directions.csv` | `individual_flow.py` | 앞 50행만(전체는 약 42,883행) |
| `individual_zone_agg.csv` | `individual_flow.py` (`aggregate_by_zone`) | 전체 |
| `bottleneck_scores.csv` | `bottleneck.py` | 전체, `risk_level`(낮음/주의/높음) 포함 |

컬럼 의미와 알려진 한계(ID 스위칭 필터링, growth/slowdown 정규화 방식 등)는 각 모듈의
docstring과 `backend/tests/`의 테스트 케이스 설명을 참고.
