# 서울 생활인구 전처리 결과

원천: 서울 생활인구(OA-23016) 행정동별 시간대 데이터 (`250_LOCAL_RESD_ADMDONG_YYYYMM/`)
스크립트: `forecast/data_pipeline/preprocess_living_pop.py`

## 산출물 (`data/processed/`)

| 파일 | 내용 |
|---|---|
| `living_pop_wide.parquet` | index=`datetime`, 컬럼=`adm_cd`(427개), 값=총생활인구. 8760행 |
| `living_pop_long.parquet` | `datetime, adm_cd, total_pop, is_imputed` (3,740,520행) |
| `quality_report.csv` | 행정동별 결측/보간/중복/0값/최대·최소·평균 |

## 범위
- 기간: 2025-11-01 00:00 ~ 2026-10-31 23:00 (1시간 간격, 8760시점)
- 행정동: 427개

## 전처리 단계
1. 월별 폴더/ZIP의 CSV 전체 읽기. `(1)` 중복 폴더는 무시.
2. 인코딩 utf-8-sig → 실패 시 cp949. 앞 4개 컬럼(날짜, 시간, 행정동, 총생활인구)만 위치로 사용.
3. 2026-07-25 이후 일부 파일의 `;` 구분자를 `,`로 치환.
4. `datetime` 생성, 인구 숫자 변환(실패 시 NaN).
5. `(datetime, adm_cd)` 중복 제거, 마지막 값 유지 (행정동당 24건 = 하루치 겹침).
6. pivot 후 전체 1시간 간격 시간축으로 reindex.
7. 연속 2시간 이하 결측만 선형보간 (`is_imputed`로 표시).

## 검증 결과
- NaN 0, 음수 0, 0값 0, 시간 간격 끊김 0, index 중복 없음
- 보간 0건 (원천에 결측 없음)
- 행정동별 최대 인구 6천~31만 → 학습 시 행정동별 스케일링 필요

## 미확인
- 이상치(급등락) 점검, 겹친 하루치의 두 값 차이
