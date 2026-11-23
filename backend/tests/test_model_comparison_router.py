from uuid import uuid4

import pytest

from app.services import vision_extended_results as extended_storage
from test_vision_extended_results import _extended_paths
from test_vision_results import bundle as bundle, client as client, seeded as seeded, upload


def _dmcount_paths(tmp_path, heatmap_bytes=b'\x89PNG\r\n\x1a\nfake'):
    heatmap = tmp_path / 'window_0.png'
    heatmap.write_bytes(heatmap_bytes)
    return _extended_paths(tmp_path, {
        'dmcount_estimates': ('window_start,zone,estimated_count,inference_ms,model_version,density_map_path\n'
                               f'0,near,6.46,10889.8,dmcount_qnrf,{heatmap}\n'),
    })


@pytest.mark.integration
def test_model_comparison_joins_yolo_and_dmcount_by_window_and_zone(db, seeded, client, bundle, tmp_path):
    assert upload(client, seeded.id, bundle).status_code == 201  # core 결과: near 구역, offset 0ms, count 3.9

    extended_storage.store_extended_result(db, seeded.id, _dmcount_paths(tmp_path))

    response = client.get(f'/api/analysis/jobs/{seeded.id}/results/model-comparison')
    assert response.status_code == 200
    payload = response.json()
    assert payload['accuracy_status'] == 'unverified'
    item = payload['items'][0]
    assert item['zone'] == 'near'
    assert item['window_start'] == 0
    # bundle의 crowd_observations는 같은 10초 윈도우 안에 sec=0(3.9), sec=1(1.7) 두 관측치가
    # 있으므로, 윈도우 평균은 (3.9+1.7)/2=2.8이어야 한다(한 시점 값을 그대로 가져오면 안 됨).
    assert item['yolo_count'] == 2.8
    assert item['dmcount_estimated_count'] == 6.46
    assert item['dmcount_model_version'] == 'dmcount_qnrf'
    assert item['density_map_path'] == f'/api/analysis/jobs/{seeded.id}/results/dmcount-estimates/density-map/window_0.png'

    image = client.get(item['density_map_path'])
    assert image.status_code == 200
    assert image.content == b'\x89PNG\r\n\x1a\nfake'


@pytest.mark.integration
def test_model_comparison_without_dmcount_is_404(db, seeded, client, bundle):
    assert upload(client, seeded.id, bundle).status_code == 201
    response = client.get(f'/api/analysis/jobs/{seeded.id}/results/model-comparison')
    assert response.status_code == 404


@pytest.mark.integration
def test_density_map_rejects_unknown_filename(db, seeded, client, bundle, tmp_path):
    assert upload(client, seeded.id, bundle).status_code == 201
    extended_storage.store_extended_result(db, seeded.id, _dmcount_paths(tmp_path))

    response = client.get(f'/api/analysis/jobs/{seeded.id}/results/dmcount-estimates/density-map/window_999.png')
    assert response.status_code == 404


@pytest.mark.integration
def test_density_map_for_other_job_is_404(db, seeded, client, bundle, tmp_path):
    assert upload(client, seeded.id, bundle).status_code == 201
    extended_storage.store_extended_result(db, seeded.id, _dmcount_paths(tmp_path))

    response = client.get(f'/api/analysis/jobs/{uuid4()}/results/dmcount-estimates/density-map/window_0.png')
    assert response.status_code == 404
