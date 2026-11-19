from datetime import datetime, timezone
from uuid import uuid4
import pytest
from pydantic import ValidationError
from app.services.capture_sessions import CaptureData, Clip, training_rows
from test_vision_results import client, seeded, bundle, upload


def data(**kw):
    return CaptureData(name='Public square', start='2026-11-17T10:00:00+09:00',
        end='2026-11-17T11:00:00+09:00', **kw)


def test_non_event_and_clock_require_evidence():
    assert data().event_phase == 'normal'
    with pytest.raises(ValidationError):
        data(event_phase='during_event')
    with pytest.raises(ValidationError):
        data(mode='synchronized')
    with pytest.raises(ValidationError):
        Clip(camera_id='a', video_id=uuid4(), clock_verified=True)
    with pytest.raises(ValidationError):
        data(is_holiday=False)


def test_causal_features_keep_missing_values_and_no_fake_calendar():
    clip = Clip(camera_id='a', video_id=uuid4())
    metrics = [dict(video_offset_ms=i*1000, mean_person_count=i, inflow=1, outflow=0,
        motion=2, low_confidence=False) for i in range(15) if i != 6]
    before = list(training_rows(metrics, clip, data(), {}, 5))
    assert before[0]['person_count'] == 2
    assert before[1]['person_count'] is None
    assert before[1]['inflow'] is None
    assert before[2]['rolling_mean'] == 12
    assert before[2]['count_change'] is None
    assert before[0]['timestamp'] is None and before[0]['is_weekend'] is None
    metrics[-1]['mean_person_count'] = 99999
    assert list(training_rows(metrics, clip, data(), {}, 5))[0] == before[0]
    assert all(r['bottleneck_score'] is None and r['direction_count'] is None for r in before)


def test_verified_clock_uses_bucket_end_and_offset():
    clip = Clip(camera_id='a', video_id=uuid4(), capture_start='2026-11-17T10:00:00+09:00',
        clock_offset_ms=2000, clock_verified=True, clock_evidence='camera clock log')
    row = next(training_rows([dict(video_offset_ms=0)], clip, data(timing_basis='recorded'), {}, 1))
    assert row['timestamp'] == '2026-11-17T10:00:03+09:00'
    assert row['is_weekend'] is False


def test_duplicate_camera_or_source_rejected():
    video = uuid4()
    with pytest.raises(ValidationError):
        data(clips=[dict(camera_id='a', video_id=video), dict(camera_id='b', video_id=video)])
    with pytest.raises(ValidationError):
        data(clips=[dict(camera_id='a', video_id=uuid4()), dict(camera_id='a', video_id=uuid4())])


def test_venue_and_session_isolation_revision_conflict(client):
    first = client.post('/api/capture/venues', json={'name': 'Square', 'venue_type': 'square'})
    second = client.post('/api/capture/venues', json={'name': 'Station', 'venue_type': 'public_facility'})
    assert first.status_code == second.status_code == 201
    a, b = first.json()['venue_id'], second.json()['venue_id']
    payload = {'venue_revision': 1, 'data': data().model_dump(mode='json')}
    created = client.post(f'/api/capture/venues/{a}/sessions', json=payload)
    assert created.status_code == 201, created.text
    sid = created.json()['session_id']
    assert client.get(f'/api/capture/venues/{b}/sessions/{sid}').status_code == 404
    assert client.get(f'/api/capture/venues/{b}/sessions').json() == []
    assert client.post(f'/api/capture/venues/{a}/sessions/{sid}/revisions', json=payload).status_code == 409
    payload['base_revision'] = 1
    assert client.post(f'/api/capture/venues/{a}/sessions/{sid}/revisions', json=payload).status_code == 201
    response = client.get(f'/api/capture/venues/{a}/sessions/{sid}/dataset.csv?interval=5')
    assert response.status_code == 200 and response.headers['x-capture-rows'] == '0'
    assert client.get(f'/api/capture/venues/{a}/sessions/{sid}/dataset.csv?interval=7').status_code == 422


def test_export_reuses_real_stored_metrics_without_modifying_result(client, seeded, bundle):
    import csv, io
    assert upload(client, seeded.id, bundle).status_code == 201
    before = client.get(f'/api/analysis/jobs/{seeded.id}/results').json()
    venue = client.post('/api/capture/venues', json={'name': 'Non-event facility'}).json()['venue_id']
    clip = dict(camera_id='one', video_id=str(seeded.video_id), job_id=str(seeded.id), roi_name='near')
    payload = dict(venue_revision=1, data=data(clips=[clip]).model_dump(mode='json'))
    response = client.post(f'/api/capture/venues/{venue}/sessions', json=payload)
    assert response.status_code == 201, response.text
    sid = response.json()['session_id']
    root = f'/api/capture/venues/{venue}/sessions/{sid}'
    output = client.get(root+'/dataset.csv?interval=1&camera_id=one')
    rows = list(csv.DictReader(io.StringIO(output.content.decode('utf-8-sig'))))
    assert len(rows) == 2 and float(rows[0]['person_count']) == 3.9
    assert rows[0]['timestamp'] == '' and rows[1]['motion'] == ''
    assert rows[0]['bottleneck_score'] == ''
    assert client.get(root+'/timeline').json()['cameras'][0]['start_seconds'] is None
    assert client.post(root+'/concentration', json=dict(session_revision=1, offset_seconds=1)).status_code == 422
    another = client.post('/api/capture/venues', json={'name': 'Other location'}).json()['venue_id']
    assert client.post(f'/api/capture/venues/{another}/sessions', json=payload).status_code == 409
    assert client.get(f'/api/analysis/jobs/{seeded.id}/results').json() == before


@pytest.mark.parametrize('count', [1, 3, 5])
def test_dynamic_cameras_before_footage_and_target_policies(client, count):
    from routers.concentration import select_target
    from routers.venues import Configuration
    config = Configuration(name='Generic', cameras=[dict(camera_id=f'c{i}', name=f'Camera {i}', point=[0,0]) for i in range(count)])
    venue = 'qa-preregister-' + uuid4().hex
    response = client.post(f'/api/venues/{venue}/revisions', json=dict(base_revision=0, configuration=config.model_dump(mode='json')))
    assert response.status_code == 201, response.text
    assert len(response.json()['configuration']['cameras']) == count
    assert select_target(config.model_dump(mode='json'), None)['cameras'] == []


def test_targets_are_selected_without_mutating_other_policies():
    from test_concentration import config_and_contexts
    from routers.concentration import select_target
    config, _ = config_and_contexts()
    config['target_policies'] = {'a': {**config['concentration_policy'], 'center_observer_camera_id': None}}
    selected = select_target(config, 'a')
    assert selected['center_zone_id'] == 'a' and selected['corridors'] == []
    assert config['center_zone_id'] == 'center' and len(config['corridors']) == 1


def test_raw_upload_limit_and_failed_commit_only_remove_new_file(client, tmp_path, monkeypatch):
    from routers import capture_uploads
    from app.database import get_db
    from app.main import app
    from unittest.mock import Mock
    from sqlalchemy.exc import SQLAlchemyError
    monkeypatch.setattr(capture_uploads, 'UPLOAD_DIR', tmp_path)
    monkeypatch.setattr(capture_uploads, 'MAX_BYTES', 10)
    monkeypatch.setattr(capture_uploads, 'RESERVE_BYTES', 0)
    monkeypatch.setattr(capture_uploads, 'probe', lambda path: 1900000)
    old = tmp_path/'existing.mp4'
    old.write_bytes(b'original')
    assert client.post('/api/capture/videos/upload?filename=long.mp4', content=b'x'*11).status_code == 413
    assert client.post('/api/capture/videos/upload?filename=bad.exe', content=b'x').status_code == 400
    prior = app.dependency_overrides[get_db]
    broken = Mock()
    broken.commit.side_effect = SQLAlchemyError('fixture failure')
    app.dependency_overrides[get_db] = lambda: broken
    try:
        assert client.post('/api/capture/videos/upload?filename=long.mp4', content=b'123').status_code == 503
    finally:
        app.dependency_overrides[get_db] = prior
    assert old.read_bytes() == b'original' and list(tmp_path.iterdir()) == [old]


def test_raw_upload_success_reuses_video_registry_without_starting_analysis(client, db, tmp_path, monkeypatch):
    from routers import capture_uploads
    from app.models import Video, AnalysisJob
    from sqlalchemy import select, func
    from uuid import UUID
    monkeypatch.setattr(capture_uploads, 'UPLOAD_DIR', tmp_path)
    monkeypatch.setattr(capture_uploads, 'RESERVE_BYTES', 0)
    monkeypatch.setattr(capture_uploads, 'probe', lambda path: None)
    before = db.scalar(select(func.count()).select_from(AnalysisJob))
    response = client.post('/api/capture/videos/upload?filename=unit-fixture.mp4', content=b'unit-fixture-only')
    assert response.status_code == 201, response.text
    row = db.get(Video, UUID(response.json()['video_id']))
    assert row.file_size == 17 and row.duration_ms is None
    assert (tmp_path/row.saved_filename).read_bytes() == b'unit-fixture-only'
    assert db.scalar(select(func.count()).select_from(AnalysisJob)) == before
    assert response.json()['analysis_status'] == 'not_requested'


def test_hybrid_contract_rejects_different_prediction_scopes():
    from app.services.forecast_contract import ForecastContract, validate_hybrid_compatibility
    contract = ForecastContract(model_version='fixture', artifact_sha256='a'*64, input_columns=['person_count'],
        target_kind='venue_zone', target_zone_id='a', horizons_seconds=[10,30], training_unit='persons',
        camera_or_zone_identifiers=['a'], minimum_history_seconds=30, python_version='fixture',
        package_versions={'test':'fixture'}, preprocessing_version='fixture', validation_metrics={'unvalidated':True}, example_output={'value':1})
    assert validate_hybrid_compatibility(contract, contract)['enabled'] is False
    with pytest.raises(ValueError):
        validate_hybrid_compatibility(contract, contract.model_copy(update={'target_zone_id':'b'}))


@pytest.mark.parametrize('count', [1, 3, 5])
def test_multiple_video_registration_and_verified_timeline_metadata(client, db, count):
    """Synthetic metadata verifies alignment logic, not real simultaneous footage."""
    from app.models import Video
    venue = client.post('/api/capture/venues', json={'name': 'Synthetic alignment QA'}).json()['venue_id']
    clips = []
    for n in range(count):
        uid = uuid4()
        db.add(Video(id=uid, filename=f'synthetic-{n}.mp4', saved_filename=f'{uid.hex}.mp4',
            storage_path=f'uploads/videos/{uid.hex}.mp4', file_size=0, duration_ms=30000))
        clips.append(dict(camera_id=f'c{n}', video_id=str(uid), capture_start='2026-11-17T10:00:00+09:00',
            clock_offset_ms=n*2000, clock_verified=True, clock_evidence='synthetic alignment test only'))
    db.flush()
    payload = dict(venue_revision=1,data=data(timing_basis='recorded', mode='synchronized', clips=clips).model_dump(mode='json'))
    response = client.post(f'/api/capture/venues/{venue}/sessions', json=payload)
    assert response.status_code == 201, response.text
    sid = response.json()['session_id']
    timeline = client.get(f'/api/capture/venues/{venue}/sessions/{sid}/timeline').json()
    assert [c['start_seconds'] for c in timeline['cameras']] == [n*2 for n in range(count)]
    assert [c['end_seconds'] for c in timeline['cameras']] == [n*2+30 for n in range(count)]
    assert all(c['reason'] is None for c in timeline['cameras'])
    assert response.json()['calendar_days'][0]['is_weekday'] is True
    assert len(client.get(f'/api/capture/venues/{venue}/sources').json()) >= count


def test_generic_walkway_registration_requires_no_video_but_checks_references(client):
    from test_concentration import config_and_contexts
    config, _ = config_and_contexts()
    config.update(cameras=[dict(camera_id='planned', name='Planned', point=[0,0])],
        corridors=[], center_zone_id=None, concentration_policy={},
        links=[dict(id='walk', name='Stairs', from_zone_id='a', to_zone_id='center', kind='stairs', camera_ids=['planned'])])
    venue = 'qa-links-' + uuid4().hex
    response = client.post(f'/api/venues/{venue}/revisions', json=dict(base_revision=0,configuration=config))
    assert response.status_code == 201, response.text
    assert response.json()['configuration']['links'][0]['kind'] == 'stairs'
    config['links'][0]['camera_ids'] = ['unknown']
    assert client.post(f'/api/venues/{venue}/revisions', json=dict(base_revision=1,configuration=config)).status_code == 422
