import json
from uuid import uuid4

import pytest

from app.services import video_forecast_results as svc


def _write_job(root, source='yolo'):
    folder = root / source
    folder.mkdir(parents=True)
    (folder / 'summary.json').write_text(json.dumps({
        'zones': ['A'], 'seconds': 40, 'target': 'mean10', 'iforest': 'ok',
        'backtest': {'10s/persistence': {'MAE': 1.0, 'RMSE': 1.2, 'skill_vs_persistence': 0.0, 'n': 2},
                     '10s/ridge_ar': {'MAE': 0.8, 'RMSE': 1.0, 'skill_vs_persistence': 0.2, 'n': 2}},
        'level_seconds': {'A': {'낮음': 37, '주의': 3}},
        'minutes': {'model_version': 'v', 'alarms': 0, 'backtest': {}, 'note': 'n'},
    }, ensure_ascii=False), encoding='utf-8')
    (folder / 'forecast_video.csv').write_text(
        '﻿zone_id,model,horizon_s,target,split,last_observed_sec,target_sec,last_observed_at,target_at,value,actual,status,shap_top\n'
        'A,persistence,10,mean10,backtest,20,30,,,3.0,4.0,ok,\n'
        'A,ridge_ar,10,mean10,backtest,20,30,,,3.5,4.0,ok,\n'
        'A,lightgbm,10,mean10,backtest,20,30,,,,4.0,insufficient_history,\n'
        'A,moving_avg_5s,10,mean10,backtest,20,30,,,9.0,4.0,ok,\n'
        'A,persistence,30,mean10,forecast,39,69,,,3.0,,ok,\n', encoding='utf-8')
    (folder / 'anomaly_timeline.csv').write_text(
        '﻿zone_id,sec,level,level_num\n' + ''.join(f'A,{s},x,{1 if 5 <= s <= 7 else 0}\n' for s in range(10)), encoding='utf-8')
    (folder / 'anomaly_events.csv').write_text(
        '﻿job_id,zone_id,event_at,event_sec,end_sec,score,method,features,level,duration_s,peak_sec,reason\n'
        'j,A,,5,7,1,level,{},주의,3.0,6,이동량 -60%\n'
        'j,A,,1,1,0.9,iforest,{},,,,\n', encoding='utf-8')


@pytest.fixture
def job(tmp_path, monkeypatch):
    job_id = uuid4()
    monkeypatch.setattr(svc, 'forecast_root', lambda _job: tmp_path)
    _write_job(tmp_path)
    return job_id


def test_series_keeps_one_horizon_and_blanks_non_ok(job):
    data = svc.video_forecast(job, 'yolo', 10)
    assert data['sources'] == ['yolo']
    assert data['series'] == [{'zone': 'A', 'sec': 30, 'split': 'backtest', 'actual': 4.0,
                               'persistence': 3.0, 'ridge_ar': 3.5, 'lightgbm': None}]
    assert [(r['model'], r['skill']) for r in data['backtest']] == [('persistence', 0.0), ('ridge_ar', 0.2)]


def test_levels_are_compressed_and_only_level_events_returned(job):
    data = svc.video_forecast(job, 'yolo', 10)
    assert data['level_segments'] == [{'zone': 'A', 'start_sec': 5, 'end_sec': 7, 'level_num': 1, 'level': '주의'}]
    assert len(data['level_events']) == 1 and data['level_events'][0]['peak_sec'] == 6


def test_missing_source_or_bad_horizon(job, tmp_path):
    with pytest.raises(svc.ForecastNotFound):
        svc.video_forecast(job, 'dmcount', 10)
    with pytest.raises(ValueError):
        svc.video_forecast(job, 'yolo', 5)


def test_no_forecast_at_all(tmp_path, monkeypatch):
    monkeypatch.setattr(svc, 'forecast_root', lambda _job: tmp_path / 'none')
    with pytest.raises(svc.ForecastNotFound):
        svc.video_forecast(uuid4(), 'yolo', 10)
