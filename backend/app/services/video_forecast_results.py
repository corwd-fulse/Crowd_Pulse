"""분석 job의 시계열 예측 결과(forecast/run_job_forecast.py 출력)를 대시보드용으로 읽는다.

위치: uploads/results/<job_id>/forecast/<source>/ (source = yolo | dmcount | fused, 구역 인원 출처; fused = YOLO 인원 + 같은 10초 창의 DM-Count 잔여(YOLO 박스 밖))
  summary.json        모델별 backtest MAE·skill, 분 단위 예측 요약
  forecast_video.csv  10·30·60초 뒤 예측(backtest + 영상 끝 이후 forecast)
  anomaly_timeline.csv / anomaly_events.csv  이상 흐름 3단계
저장: import_run()이 위 파일을 DB로 옮긴다 - 예측 점은 crowd_predictions(count_source, model_name, horizon_s 등),
요약과 경보 단계는 video_forecast_runs(job x 출처당 1행). 같은 job과 출처를 다시 넣으면 한 트랜잭션에서 교체한다.
조회: DB에 있으면 DB, 없으면 예전처럼 파일(이관 전 결과 호환). 행정동 생활인구 예측(population-forecast API)과는 테이블이 섞이지 않는다.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.analytics import CrowdPrediction, VideoForecastRun

from app.services.vision_results import BASE_DIR

SOURCES = ('yolo', 'dmcount', 'fused')
HORIZONS = (10, 30, 60)
SERIES_MODELS = ('persistence', 'ridge_ar', 'lightgbm', 'lightgbm_pre', 'lstm_pre')
LEVEL_NAMES = {0: '낮음', 1: '주의', 2: '높음'}


class ForecastNotFound(Exception):
    pass


def forecast_root(job_id: UUID) -> Path:
    return (BASE_DIR / 'uploads/results' / str(job_id) / 'forecast').resolve()


def available_sources(job_id: UUID) -> list[str]:
    root = forecast_root(job_id)
    return [s for s in SOURCES if (root / s / 'summary.json').is_file() and (root / s / 'forecast_video.csv').is_file()]


def _num(text: str) -> float | None:
    if text in ('', None):
        return None
    value = float(text)
    return None if math.isnan(value) else value


def _read_csv(path: Path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        yield from csv.DictReader(handle)


def _backtest(summary: dict) -> list[dict]:
    rows = []
    for key, stats in summary.get('backtest', {}).items():
        horizon, model = key.split('/', 1)
        rows.append({'horizon_s': int(horizon.rstrip('s')), 'model': model, 'mae': stats.get('MAE'), 'rmse': stats.get('RMSE'),
                     'skill': stats.get('skill_vs_persistence'), 'n': int(stats.get('n') or 0)})
    return sorted(rows, key=lambda r: (r['horizon_s'], r['model']))


def _series(path: Path, horizon: int) -> list[dict]:
    """target_sec 기준 한 행에 모델별 예측값과 실제값. status != ok 인 예측은 비운다."""
    points: dict[tuple[str, int], dict] = {}
    for row in _read_csv(path):
        if int(row['horizon_s']) != horizon or row['model'] not in SERIES_MODELS:
            continue
        key = (row['zone_id'], int(row['target_sec']))
        point = points.setdefault(key, {'zone': key[0], 'sec': key[1], 'split': row['split'], 'actual': _num(row['actual'])})
        point[row['model']] = _num(row['value']) if row['status'] == 'ok' else None
    return sorted(points.values(), key=lambda p: (p['zone'], p['sec']))


def _level_segments(path: Path) -> list[dict]:
    """매초 단계를 같은 단계가 이어지는 구간으로 압축한다(낮음 구간은 생략)."""
    segments: list[dict] = []
    current = None
    for row in _read_csv(path):
        zone, sec, level = row['zone_id'], int(row['sec']), int(float(row['level_num']))
        if current and current['zone'] == zone and current['level_num'] == level and current['end_sec'] == sec - 1:
            current['end_sec'] = sec
            continue
        if current and current['level_num'] > 0:
            segments.append(current)
        current = {'zone': zone, 'start_sec': sec, 'end_sec': sec, 'level_num': level, 'level': LEVEL_NAMES.get(level, str(level))}
    if current and current['level_num'] > 0:
        segments.append(current)
    return segments


def _level_events(path: Path) -> list[dict]:
    events = []
    for row in _read_csv(path):
        if row['method'] != 'level':
            continue
        events.append({'zone': row['zone_id'], 'start_sec': int(float(row['event_sec'])), 'end_sec': int(float(row['end_sec'])),
                       'peak_sec': int(float(row['peak_sec'])) if row['peak_sec'] else None, 'level': row['level'],
                       'duration_s': _num(row['duration_s']), 'reason': row['reason']})
    return sorted(events, key=lambda e: (e['start_sec'], e['zone']))


def _response(job_id: UUID, source: str, sources: list[str], horizon: int, summary: dict, series: list[dict],
              segments: list[dict], events: list[dict], stored: str) -> dict:
    minutes = summary.get('minutes') or {}
    return {
        'analysis_job_id': str(job_id),
        'source': source,
        'sources': sources,
        'stored': stored,
        'horizon_s': horizon,
        'horizons': list(HORIZONS),
        'zones': summary.get('zones', []),
        'seconds': summary.get('seconds'),
        'target': summary.get('target', 'mean10'),
        'models': list(SERIES_MODELS),
        'backtest': _backtest(summary),
        'series': series,
        'level_segments': segments,
        'level_seconds': summary.get('level_seconds', {}),
        'level_events': events,
        'iforest': summary.get('iforest'),
        'minutes': {'model_version': minutes.get('model_version'), 'alarms': minutes.get('alarms', 0),
                    'backtest': minutes.get('backtest', {}), 'note': minutes.get('note')} if minutes else None,
        'note': ('예측값은 10초 평균 인원(mean10)이며, 걷기식 backtest(그 시각까지의 영상만 사용)로 평가했습니다. '
                 'skill > 0 이면 직전값(persistence)보다 오차가 작습니다. 이상 흐름 단계는 참고 지표입니다.'),
    }


def stored_sources(db: Session, job_id: UUID) -> list[str]:
    found = set(db.scalars(select(VideoForecastRun.count_source).where(VideoForecastRun.analysis_job_id == job_id)))
    return [s for s in SOURCES if s in found]


def _series_from_db(db: Session, job_id: UUID, source: str, horizon: int) -> list[dict]:
    rows = db.execute(select(CrowdPrediction.region_key, CrowdPrediction.target_offset_ms, CrowdPrediction.model_name,
                             CrowdPrediction.predicted_value, CrowdPrediction.actual_count, CrowdPrediction.split)
                      .where(CrowdPrediction.analysis_job_id == job_id, CrowdPrediction.count_source == source,
                             CrowdPrediction.horizon_s == horizon, CrowdPrediction.model_name.in_(SERIES_MODELS)))
    points: dict[tuple[str, int], dict] = {}
    for zone, target_ms, model, value, actual, split in rows:
        key = (zone, int(target_ms // 1000))
        point = points.setdefault(key, {'zone': zone, 'sec': key[1], 'split': split, 'actual': actual})
        point[model] = value
    return sorted(points.values(), key=lambda p: (p['zone'], p['sec']))


def video_forecast(job_id: UUID, source: str, horizon: int, db: Session | None = None) -> dict:
    in_db = stored_sources(db, job_id) if db is not None else []
    in_files = available_sources(job_id)
    sources = [s for s in SOURCES if s in in_db or s in in_files]
    if not sources:
        raise ForecastNotFound('Time-series forecast was not run for this analysis job')
    if source not in sources:
        raise ForecastNotFound(f'No forecast for source {source!r} (available: {sources})')
    if horizon not in HORIZONS:
        raise ValueError(f'horizon must be one of {HORIZONS}')
    if source in in_db:
        run = db.scalars(select(VideoForecastRun).where(VideoForecastRun.analysis_job_id == job_id,
                                                        VideoForecastRun.count_source == source)).one()
        return _response(job_id, source, sources, horizon, run.summary, _series_from_db(db, job_id, source, horizon),
                         run.level_segments, run.level_events, 'db')
    folder = forecast_root(job_id) / source
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    timeline, events = folder / 'anomaly_timeline.csv', folder / 'anomaly_events.csv'
    return _response(job_id, source, sources, horizon, summary, _series(folder / 'forecast_video.csv', horizon),
                     _level_segments(timeline) if timeline.is_file() else [],
                     _level_events(events) if events.is_file() else [], 'file')


def import_run(db: Session, job_id: UUID, source: str, root: Path | None = None, provenance: dict | None = None) -> dict:
    """<root>/<source>/ 의 예측 파일을 DB로 옮긴다. 같은 job과 출처의 이전 결과는 같은 트랜잭션에서 지우고 다시 넣는다
    (재실행해도 중복 없음). commit은 호출한 쪽이 한다. 파일은 지우지 않는다."""
    if source not in SOURCES:
        raise ValueError(f'source must be one of {SOURCES}')
    folder = Path(root or forecast_root(job_id)) / source
    if not ((folder / 'summary.json').is_file() and (folder / 'forecast_video.csv').is_file()):
        raise ForecastNotFound(f'No forecast files in {folder}')
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    timeline, events_csv = folder / 'anomaly_timeline.csv', folder / 'anomaly_events.csv'
    segments = _level_segments(timeline) if timeline.is_file() else []
    events = _level_events(events_csv) if events_csv.is_file() else []
    predicted_at = datetime.now(timezone.utc)
    points, skipped = [], defaultdict(int)
    for row in _read_csv(folder / 'forecast_video.csv'):
        value = _num(row['value'])
        if row['status'] != 'ok' or value is None:
            # 학습 이력 부족 등으로 값이 없는 예측은 저장하지 않는다(0명으로 대체하지 않음)
            skipped[row['status'] or 'no_value'] += 1
            continue
        points.append({
            'id': uuid4(), 'predicted_at': predicted_at, 'analysis_job_id': job_id, 'region_key': row['zone_id'],
            'source_offset_ms': int(row['last_observed_sec']) * 1000, 'target_offset_ms': int(row['target_sec']) * 1000,
            'target_captured_at': datetime.fromisoformat(row['target_at']) if row.get('target_at') else None,
            'predicted_person_count': max(0, round(value)), 'predicted_congestion_percent': None,
            'model_version': f"video-forecast/{row['model']}/{row['target']}", 'count_source': source,
            'model_name': row['model'], 'horizon_s': int(row['horizon_s']), 'split': row['split'],
            # 모델 원래 값 그대로(ridge_ar 등은 음수도 낸다). 0 이상 제약은 정수 칸(predicted_person_count)에만 둔다.
            'predicted_value': round(value, 3), 'actual_count': _num(row['actual']), 'status': row['status'],
        })
    db.execute(delete(CrowdPrediction).where(CrowdPrediction.analysis_job_id == job_id, CrowdPrediction.count_source == source))
    db.execute(delete(VideoForecastRun).where(VideoForecastRun.analysis_job_id == job_id, VideoForecastRun.count_source == source))
    db.flush()
    for i in range(0, len(points), 5000):
        db.execute(CrowdPrediction.__table__.insert(), points[i:i + 5000])
    db.add(VideoForecastRun(analysis_job_id=job_id, count_source=source, predicted_at=predicted_at, summary=summary,
                            level_segments=segments, level_events=events, point_count=len(points),
                            provenance={**(provenance or {}), 'folder': str(folder), 'skipped_points': dict(skipped)}))
    db.flush()
    return {'job_id': str(job_id), 'source': source, 'points': len(points), 'skipped': dict(skipped),
            'level_events': len(events), 'level_segments': len(segments)}
