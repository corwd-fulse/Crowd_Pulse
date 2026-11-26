import json

import pytest

from app.services.vision_extended_csv import InvalidExtendedResult, validate_extended_bundle

CONTENTS = {
    'dwell_times': 'track_id,zone,enter_sec,exit_sec,dwell_s\n1,near,0,2,2.0\n',
    'counter_flow': 'sec,zone,mean_angle_deg,reverse_ratio,n_tracks\n0,near,0.0,0.5,2\n1,near,10.0,,0\n',
    'queue_scores': 'window_start,zone,queue_score,density,stillness,is_queue\n0,near,80.0,0.9,0.9,True\n',
    'density_heatmap': 'window_start,cell_x,cell_y,count\n0,3,4,1.5\n',
    'tracking_quality': json.dumps({
        'track_count': 1, 'mean_track_length_frames': 20.0, 'median_track_length_frames': 20.0,
        'mean_confidence': 0.9, 'min_confidence': 0.8, 'suspected_id_switch_ratio': 0.0,
    }),
}


def _write(tmp_path, overrides=None):
    contents = {**CONTENTS, **(overrides or {})}
    paths = {}
    for kind, text in contents.items():
        suffix = '.json' if kind == 'tracking_quality' else '.csv'
        path = tmp_path / f'{kind}{suffix}'
        path.write_text(text, encoding='utf-8')
        paths[kind] = path
    return paths


def test_valid_bundle_parses_all_kinds(tmp_path):
    paths = _write(tmp_path)
    manifest, records = validate_extended_bundle(paths)
    assert set(manifest) == set(CONTENTS)
    assert records['dwell_times'][0]['dwell_s'] == 2.0
    assert records['counter_flow'][1]['reverse_ratio'] is None
    assert records['tracking_quality']['track_count'] == 1


def test_missing_required_file_is_rejected(tmp_path):
    paths = _write(tmp_path)
    del paths['queue_scores']
    with pytest.raises(InvalidExtendedResult):
        validate_extended_bundle(paths)


def test_malformed_header_is_rejected(tmp_path):
    paths = _write(tmp_path, {'dwell_times': 'track_id,zone,dwell_s\n1,near,2.0\n'})
    with pytest.raises(InvalidExtendedResult):
        validate_extended_bundle(paths)


def test_out_of_range_ratio_is_rejected(tmp_path):
    paths = _write(tmp_path, {'queue_scores': 'window_start,zone,queue_score,density,stillness,is_queue\n0,near,80.0,1.5,0.9,True\n'})
    with pytest.raises(InvalidExtendedResult):
        validate_extended_bundle(paths)


def test_exit_before_enter_is_rejected(tmp_path):
    paths = _write(tmp_path, {'dwell_times': 'track_id,zone,enter_sec,exit_sec,dwell_s\n1,near,5,2,2.0\n'})
    with pytest.raises(InvalidExtendedResult):
        validate_extended_bundle(paths)


def test_speed_files_are_accepted_when_present(tmp_path):
    paths = _write(tmp_path, {
        'crowd_speed_zone': 'sec,zone,speed_mps,low_confidence\n0,near,1.2,False\n',
        'crowd_speed_individual': 'frame,t_ms,track_id,speed_mps,reliable\n1,33,1,1.1,True\n',
    })
    manifest, records = validate_extended_bundle(paths)
    assert records['crowd_speed_zone'][0]['speed_mps'] == 1.2


def test_individual_speed_is_validated_but_kept_file_only(tmp_path):
    paths = _write(tmp_path, {
        'crowd_speed_zone': 'sec,zone,speed_mps,low_confidence\n0,near,1.2,False\n',
        'crowd_speed_individual': 'frame,t_ms,track_id,speed_mps,reliable\n1,33,1,1.1,True\n2,66,1,1.0,True\n',
    })
    manifest, records = validate_extended_bundle(paths)
    assert 'crowd_speed_individual' not in records  # DB JSON 칸에 넣지 않는다
    assert manifest['crowd_speed_individual']['rows'] == 2
    assert manifest['crowd_speed_individual']['file_only'] is True


def test_file_only_individual_speed_still_rejects_bad_rows(tmp_path):
    paths = _write(tmp_path, {
        'crowd_speed_individual': 'frame,t_ms,track_id,speed_mps,reliable\n1,33,1,-1,True\n',
    })
    with pytest.raises(InvalidExtendedResult):
        validate_extended_bundle(paths)


def test_file_only_individual_speed_is_not_limited_to_64mib(tmp_path, monkeypatch):
    from app.services import vision_extended_csv
    monkeypatch.setattr(vision_extended_csv, 'MAX_FILE_BYTES', 10)  # 일반 종류 상한을 아주 작게
    paths = _write(tmp_path, {
        'crowd_speed_individual': 'frame,t_ms,track_id,speed_mps,reliable\n1,33,1,1.1,True\n',
    })
    paths = {k: v for k, v in paths.items() if k in ('crowd_speed_individual',)}
    assert vision_extended_csv.count_valid_csv_rows(paths['crowd_speed_individual'], 'crowd_speed_individual') == 1
