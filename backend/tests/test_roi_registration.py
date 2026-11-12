from copy import deepcopy

from test_vision_results import client as client, seeded as seeded, bundle as bundle


def test_conditional_roi_registration_roundtrip_and_preservation(client, seeded, bundle):
    _, roi, _ = bundle
    snapshot = deepcopy(roi)
    snapshot.update(status='candidate', approved_for_analysis=False)
    url = f'/api/videos/{seeded.video_id}/roi'
    assert client.get(url).status_code == 404
    response = client.put(url, json=snapshot, headers={'If-None-Match': '*'})
    assert response.status_code == 200
    assert response.json()['roi_status'] == 'candidate'
    assert client.get(url).json()['roi_snapshot'] == snapshot
    replacement = deepcopy(snapshot)
    replacement['zones'][0]['name'] = 'changed'
    assert client.put(url, json=replacement, headers={'If-None-Match': '*'}).status_code == 412
    assert client.get(url).json()['roi_snapshot'] == snapshot


def test_invalid_condition_does_not_store(client, seeded, bundle):
    _, roi, _ = bundle
    url = f'/api/videos/{seeded.video_id}/roi'
    assert client.put(url, json=roi, headers={'If-None-Match': 'other'}).status_code == 400
    assert client.get(url).status_code == 404
