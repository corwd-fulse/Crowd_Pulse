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


# --- DB 저장(import_run) --------------------------------------------------------------

@pytest.fixture
def db_job(db, tmp_path, monkeypatch):
    from app.models import AnalysisJob, Video
    uid = uuid4()
    video = Video(id=uid, filename='fixture.mp4', saved_filename=f'{uid.hex}.mp4',
                  storage_path=f'uploads/videos/{uid.hex}.mp4', file_size=5)
    db.add(video)
    db.flush()
    job = AnalysisJob(video_id=video.id)
    db.add(job)
    db.flush()
    monkeypatch.setattr(svc, 'forecast_root', lambda _job: tmp_path)
    _write_job(tmp_path)
    return job.id


def _count(db, job_id):
    from sqlalchemy import func, select
    from app.models import CrowdPrediction, VideoForecastRun
    return (db.scalar(select(func.count()).select_from(CrowdPrediction).where(CrowdPrediction.analysis_job_id == job_id)),
            db.scalar(select(func.count()).select_from(VideoForecastRun).where(VideoForecastRun.analysis_job_id == job_id)))


def test_import_stores_ok_points_and_reads_back_like_files(db, db_job):
    from_files = svc.video_forecast(db_job, 'yolo', 10, db)
    assert from_files['stored'] == 'file'
    result = svc.import_run(db, db_job, 'yolo')
    assert result['points'] == 4 and result['skipped'] == {'insufficient_history': 1}
    from_db = svc.video_forecast(db_job, 'yolo', 10, db)
    assert from_db['stored'] == 'db'
    for key in from_files:
        if key not in ('stored', 'series'):
            assert from_db[key] == from_files[key], key
    # 파일은 값 없는 lightgbm을 None으로 두고, DB는 그 모델 행 자체를 저장하지 않는다
    assert from_db['series'] == [{'zone': 'A', 'sec': 30, 'split': 'backtest', 'actual': 4.0, 'persistence': 3.0, 'ridge_ar': 3.5}]


def test_reimport_replaces_instead_of_duplicating(db, db_job):
    svc.import_run(db, db_job, 'yolo')
    first = _count(db, db_job)
    svc.import_run(db, db_job, 'yolo')
    assert _count(db, db_job) == first == (4, 1)


def test_negative_model_output_is_kept_exactly(db, db_job, tmp_path):
    csv = tmp_path / 'yolo' / 'forecast_video.csv'
    csv.write_text(csv.read_text(encoding='utf-8').replace('A,ridge_ar,10,mean10,backtest,20,30,,,3.5', 'A,ridge_ar,10,mean10,backtest,20,30,,,-1.25'), encoding='utf-8')
    svc.import_run(db, db_job, 'yolo')
    from sqlalchemy import select
    from app.models import CrowdPrediction
    row = db.scalars(select(CrowdPrediction).where(CrowdPrediction.analysis_job_id == db_job, CrowdPrediction.model_name == 'ridge_ar')).one()
    assert row.predicted_value == -1.25 and row.predicted_person_count == 0 and row.predicted_congestion_percent is None
    assert svc.video_forecast(db_job, 'yolo', 10, db)['series'][0]['ridge_ar'] == -1.25


def test_import_rejects_unknown_source_and_missing_files(db, db_job):
    with pytest.raises(ValueError):
        svc.import_run(db, db_job, 'other')
    with pytest.raises(svc.ForecastNotFound):
        svc.import_run(db, db_job, 'dmcount')
