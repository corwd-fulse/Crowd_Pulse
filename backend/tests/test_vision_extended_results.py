import json

import pytest

from app.models import VisionExtendedResult
from app.services import vision_extended_results as extended_storage
from app.services import vision_results
from app.services.vision_extended_csv import InvalidExtendedResult
from app.services.vision_results import JobNotFound, ResultConflict
from test_vision_results import bundle as bundle, seeded as seeded, client as client

CONTENTS = {
    'dwell_times': 'track_id,zone,enter_sec,exit_sec,dwell_s\n1,near,0,2,2.0\n',
    'counter_flow': 'sec,zone,mean_angle_deg,reverse_ratio,n_tracks\n0,near,0.0,0.5,2\n',
    'queue_scores': 'window_start,zone,queue_score,density,stillness,is_queue\n0,near,80.0,0.9,0.9,True\n',
    'density_heatmap': 'window_start,cell_x,cell_y,count\n0,3,4,1.5\n',
    'tracking_quality': json.dumps({
        'track_count': 1, 'mean_track_length_frames': 20.0, 'median_track_length_frames': 20.0,
        'mean_confidence': 0.9, 'min_confidence': 0.8, 'suspected_id_switch_ratio': 0.0,
    }),
}


def _extended_paths(tmp_path, overrides=None):
    contents = {**CONTENTS, **(overrides or {})}
    paths = {}
    for kind, text in contents.items():
        suffix = '.json' if kind == 'tracking_quality' else '.csv'
        path = tmp_path / f'{kind}{suffix}'
        path.write_text(text, encoding='utf-8')
        paths[kind] = path
    return paths


@pytest.mark.integration
def test_extended_result_requires_core_result_first(db, seeded, tmp_path):
    paths = _extended_paths(tmp_path)
    with pytest.raises(JobNotFound):
        extended_storage.store_extended_result(db, seeded.id, paths)


@pytest.mark.integration
def test_extended_result_stores_and_rejects_duplicate(db, seeded, bundle, tmp_path):
    core_paths, roi, meta = bundle
    vision_results.store_result(db, seeded.id, core_paths, roi, meta)

    paths = _extended_paths(tmp_path)
    stored = extended_storage.store_extended_result(db, seeded.id, paths)
    assert stored.analysis_job_id == seeded.id
    assert stored.dwell_times[0]['dwell_s'] == 2.0
    assert stored.pixels_per_meter is None
    assert stored.crowd_speed_zone is None

    fetched = extended_storage.extended_result_for_job(db, seeded.id)
    assert isinstance(fetched, VisionExtendedResult)

    with pytest.raises(ResultConflict):
        extended_storage.store_extended_result(db, seeded.id, paths)


@pytest.mark.integration
def test_speed_file_requires_calibration_value(db, seeded, bundle, tmp_path):
    core_paths, roi, meta = bundle
    vision_results.store_result(db, seeded.id, core_paths, roi, meta)

    paths = _extended_paths(tmp_path, {
        'crowd_speed_zone': 'sec,zone,speed_mps,low_confidence\n0,near,1.2,False\n',
    })
    with pytest.raises(InvalidExtendedResult):
        extended_storage.store_extended_result(db, seeded.id, paths)  # no pixels_per_meter passed

    stored = extended_storage.store_extended_result(db, seeded.id, paths, pixels_per_meter=50.0)
    assert stored.pixels_per_meter == 50.0
    assert stored.crowd_speed_zone[0]['speed_mps'] == 1.2
