"""분석 job의 시계열 예측 결과(forecast/run_job_forecast.py 출력)를 대시보드용으로 읽는다.

위치: uploads/results/<job_id>/forecast/<source>/ (source = yolo | dmcount | fused, 구역 인원 출처; fused = YOLO 인원 + 같은 10초 창의 DM-Count 잔여(YOLO 박스 밖))
  summary.json        모델별 backtest MAE·skill, 분 단위 예측 요약
  forecast_video.csv  10·30·60초 뒤 예측(backtest + 영상 끝 이후 forecast)
  anomaly_timeline.csv / anomaly_events.csv  이상 흐름 3단계
예측 결과는 DB에 넣지 않은 오프라인 산출물이라 파일을 그대로 읽는다(표준 라이브러리만 사용).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from uuid import UUID

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


def video_forecast(job_id: UUID, source: str, horizon: int) -> dict:
    sources = available_sources(job_id)
    if not sources:
        raise ForecastNotFound('Time-series forecast was not run for this analysis job')
    if source not in sources:
        raise ForecastNotFound(f'No forecast for source {source!r} (available: {sources})')
    if horizon not in HORIZONS:
        raise ValueError(f'horizon must be one of {HORIZONS}')
    folder = forecast_root(job_id) / source
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    minutes = summary.get('minutes') or {}
    timeline, events = folder / 'anomaly_timeline.csv', folder / 'anomaly_events.csv'
    return {
        'analysis_job_id': str(job_id),
        'source': source,
        'sources': sources,
        'horizon_s': horizon,
        'horizons': list(HORIZONS),
        'zones': summary.get('zones', []),
        'seconds': summary.get('seconds'),
        'target': summary.get('target', 'mean10'),
        'models': list(SERIES_MODELS),
        'backtest': _backtest(summary),
        'series': _series(folder / 'forecast_video.csv', horizon),
        'level_segments': _level_segments(timeline) if timeline.is_file() else [],
        'level_seconds': summary.get('level_seconds', {}),
        'level_events': _level_events(events) if events.is_file() else [],
        'iforest': summary.get('iforest'),
        'minutes': {'model_version': minutes.get('model_version'), 'alarms': minutes.get('alarms', 0),
                    'backtest': minutes.get('backtest', {}), 'note': minutes.get('note')} if minutes else None,
        'note': ('예측값은 10초 평균 인원(mean10)이며, 걷기식 backtest(그 시각까지의 영상만 사용)로 평가했습니다. '
                 'skill > 0 이면 직전값(persistence)보다 오차가 작습니다. 이상 흐름 단계는 참고 지표입니다.'),
    }
