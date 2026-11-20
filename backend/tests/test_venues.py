from copy import deepcopy
from uuid import uuid4
import pytest
from test_vision_results import client, seeded, bundle, upload
from app.models import VenueRevision


def configuration():
    return dict(name='검증 행사장', zones=[dict(id='a',name='A',polygon=[[10,10],[200,10],[200,200],[10,200]],color='#44b6c8')])


def test_append_history_and_conflict_preserve_old_configuration(client, db):
    venue = 'qa-' + uuid4().hex
    path = f'/api/venues/{venue}'
    assert client.get(path).status_code == 404
    config = configuration()
    first = client.post(path+'/revisions',json=dict(base_revision=0,configuration=config))
    assert first.status_code == 201, first.text
    config['zones'][0]['note'] = '현장 점검 검토'
    second = client.post(path+'/revisions',json=dict(base_revision=1,configuration=config))
    assert second.status_code == 201
    assert client.post(path+'/revisions',json=dict(base_revision=1,configuration=config)).status_code == 409
    assert client.get(path).json()['revision'] == 2
    assert client.get(path+'/revisions/1').json()['configuration']['zones'][0]['note'] == ''
    assert [r['revision'] for r in client.get(path+'/revisions').json()] == [2,1]


@pytest.mark.parametrize('change', ['overlap','self_intersect','bounds','duplicate','unknown_mapping','svg','capacity'])
def test_rejects_unsafe_or_ambiguous_configuration(client, change):
    config = configuration()
    zone = config['zones'][0]
    if change == 'overlap': config['zones'].append({**zone,'id':'b','name':'B'})
    if change == 'self_intersect': zone['polygon'] = [[0,0],[100,100],[0,100],[100,0]]
    if change == 'bounds': zone['polygon'][0] = [-1,0]
    if change == 'duplicate': config['zones'].append({**zone,'polygon':[[300,300],[500,300],[500,500]]})
    if change == 'unknown_mapping': zone['mapping'] = dict(job_id=str(uuid4()),roi_name='A')
    if change == 'svg': config['image'] = 'data:image/svg+xml;base64,PHN2Zz48L3N2Zz4='
    if change == 'capacity': zone['capacity'] = 0
    assert client.post('/api/venues/qa/revisions',json=dict(base_revision=0,configuration=config)).status_code == 422


def test_mapping_pins_stored_roi_without_modifying_analysis(client, seeded, bundle, db):
    assert upload(client,seeded.id,bundle).status_code == 201
    config = configuration()
    config['zones'][0]['mapping'] = dict(job_id=str(seeded.id),roi_name='near')
    venue = 'qa-'+uuid4().hex
    response = client.post(f'/api/venues/{venue}/revisions',json=dict(base_revision=0,configuration=config))
    assert response.status_code == 201, response.text
    stored = db.get(VenueRevision,(venue,1)).configuration
    snapshot = deepcopy(stored['mapping_roi_snapshots'][str(seeded.id)])
    assert snapshot == bundle[1]
    result_before = client.get(f'/api/analysis/jobs/{seeded.id}/results').json()
    config['zones'][0]['name'] = '운영 구역 이름 변경'
    assert client.post(f'/api/venues/{venue}/revisions',json=dict(base_revision=1,configuration=config)).status_code == 201
    assert client.get(f'/api/analysis/jobs/{seeded.id}/results').json() == result_before
    assert db.get(VenueRevision,(venue,1)).configuration['mapping_roi_snapshots'][str(seeded.id)] == snapshot
