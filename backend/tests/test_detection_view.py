import json
from uuid import uuid4

import pytest

from app.models import VisionResult
from app.services import vision_results as storage
from test_vision_results import client as client, seeded as seeded, bundle as bundle, upload


@pytest.fixture
def tracking(client, seeded, bundle, db):
    assert upload(client, seeded.id, bundle).status_code == 201
    seeded.worker_token = uuid4()
    db.flush()
    result = db.get(VisionResult, seeded.id)
    root = storage.BASE_DIR / 'uploads/analysis' / str(seeded.id) / f'{seeded.worker_token.hex}-test'
    (root/'tracker').mkdir(parents=True)
    (root/'execution_metadata.json').write_text(json.dumps(result.execution_snapshot), encoding='utf-8')
    path = root/'tracker/video_detections.csv'
    path.write_text('frame,t_ms,track_id,x1,y1,x2,y2,conf\n0,0,7,10,20,30,80,0.9\n1,40,,11,20,31,80,0.8\n', encoding='utf-8')
    return path, f'/api/analysis/jobs/{seeded.id}/detections'


def test_tracking_roundtrip_and_empty_id(client, tracking):
    path, url = tracking
    response = client.get(url)
    assert response.status_code == 200
    data = response.json()
    assert data['fps'] == 25 and data['frame_count'] == 404
    assert data['video_resolution'] == [1920,1080]
    assert data['frames'][0]['boxes'][0]['track_id'] == 7
    assert data['frames'][1]['boxes'][0]['track_id'] is None
    assert str(path) not in response.text


def test_missing_or_unproven_tracking_is_not_guessed(client, tracking):
    path, url = tracking
    (path.parent.parent/'execution_metadata.json').write_text('{}', encoding='utf-8')
    assert client.get(url).status_code == 404


@pytest.mark.parametrize('row', ['1,0,7,10,20,30,80,0.9', '0,0,7,10,20,3000,80,0.9', '0,0,7,NaN,20,30,80,0.9'])
def test_corrupt_csv_is_an_error_not_empty_results(client, tracking, row):
    path, url = tracking
    path.write_text('frame,t_ms,track_id,x1,y1,x2,y2,conf\n'+row+'\n', encoding='utf-8')
    assert client.get(url).status_code == 422


def test_header_only_and_unfinished_job(client, tracking, seeded, db):
    path, url = tracking
    path.write_text('frame,t_ms,track_id,x1,y1,x2,y2,conf\n', encoding='utf-8')
    assert client.get(url).json()['frames'] == []
    seeded.status = 'running'
    db.flush()
    assert client.get(url).status_code == 404
