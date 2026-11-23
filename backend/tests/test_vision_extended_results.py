import json

import pytest

from app.models import VisionExtendedResult
from app.services import vision_extended_results as extended_storage
from app.services import vision_results
from app.services.vision_extended_csv import InvalidExtendedResult
from app.services.vision_results import JobNotFound, ResultConflict
from test_vision_results import bundle as bundle, seeded as seeded

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


def test_dict_calibration_is_rejected_before_touching_db(tmp_path):
    # pixels_per_meter은 Float 컬럼 하나뿐이라 구역별 dict 보정값은 DB에 못 담는다.
    # DB에 닿기 전에 걸러져야 하므로 db=None으로도 테스트할 수 있다.
    paths = _extended_paths(tmp_path, {
        'crowd_speed_zone': 'sec,zone,speed_mps,low_confidence\n0,near,1.2,False\n',
    })
    with pytest.raises(InvalidExtendedResult):
        extended_storage.store_extended_result(None, None, paths, pixels_per_meter={'near': 50.0})


def test_individual_speed_without_calibration_is_rejected_before_touching_db(tmp_path):
    # crowd_speed_individual만 있고 crowd_speed_zone은 없는 경우도 pixels_per_meter 없이는
    # 막혀야 한다 — 어느 파일이든 속도(m/s) 값은 보정값 기록 없이 저장되면 안 된다.
    paths = _extended_paths(tmp_path, {
        'crowd_speed_individual': 'frame,t_ms,track_id,speed_mps,reliable\n1,33,1,1.1,True\n',
    })
    with pytest.raises(InvalidExtendedResult):
        extended_storage.store_extended_result(None, None, paths)


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


@pytest.mark.integration
def test_dmcount_estimates_are_independent_of_calibration(db, seeded, bundle, tmp_path):
    # DM-Count는 YOLO+ByteTrack과 완전히 별개 모델이라 pixels_per_meter(카메라 보정) 유무와
    # 무관하게 저장돼야 한다 — crowd_speed_zone과 달리 required-pairing 검증이 없다.
    core_paths, roi, meta = bundle
    vision_results.store_result(db, seeded.id, core_paths, roi, meta)

    paths = _extended_paths(tmp_path, {
        'dmcount_estimates': ('window_start,zone,estimated_count,inference_ms,model_version,density_map_path\n'
                               '0,near,6.46,10889.8,dmcount_qnrf,\n'),
    })
    stored = extended_storage.store_extended_result(db, seeded.id, paths)  # 보정값 없이도 저장돼야 함

    assert stored.dmcount_estimates == [{
        'window_start': 0, 'zone': 'near', 'estimated_count': 6.46,
        'inference_ms': 10889.8, 'model_version': 'dmcount_qnrf', 'density_map_path': None,
    }]
    assert stored.pixels_per_meter is None


@pytest.mark.integration
def test_dmcount_density_map_image_is_copied_into_permanent_storage(db, seeded, bundle, tmp_path):
    # vision_extended_results.py는 vision_results.BASE_DIR을 `from ... import` 시점에
    # 복사해 가지므로, seeded 픽스처가 vision_results.BASE_DIR만 monkeypatch해도
    # extended_storage가 실제로 쓰는 BASE_DIR은 그대로다 — 저장 위치를 찾을 땐 그 값을 써야 한다.
    base_dir = extended_storage.BASE_DIR

    core_paths, roi, meta = bundle
    vision_results.store_result(db, seeded.id, core_paths, roi, meta)

    # density_estimation.py가 실제로 쓰는 임시 PNG를 흉내낸다 — dmcount_estimates.csv와
    # 같은 디렉터리(tmp_path) 밑이라 신뢰 경계(trusted_root) 안이다.
    heatmap_src = tmp_path / 'window_0.png'
    heatmap_src.write_bytes(b'\x89PNG\r\n\x1a\nfake-png-bytes')

    paths = _extended_paths(tmp_path, {
        'dmcount_estimates': ('window_start,zone,estimated_count,inference_ms,model_version,density_map_path\n'
                               f'0,near,6.46,10889.8,dmcount_qnrf,{heatmap_src}\n'),
    })
    stored = extended_storage.store_extended_result(db, seeded.id, paths)

    saved_relative_path = stored.dmcount_estimates[0]['density_map_path']
    assert saved_relative_path != str(heatmap_src)  # 원본 임시 경로가 아니라 BASE_DIR 상대경로로 바뀜
    saved_path = base_dir / saved_relative_path
    try:
        assert saved_path.is_file()
        assert saved_path.read_bytes() == heatmap_src.read_bytes()
    finally:
        saved_path.unlink(missing_ok=True)  # 이 테스트가 실제 repo uploads/results/ 밑에 남기는 파일 정리


@pytest.mark.integration
def test_dmcount_density_map_path_outside_trusted_root_is_rejected(db, seeded, bundle, tmp_path):
    core_paths, roi, meta = bundle
    vision_results.store_result(db, seeded.id, core_paths, roi, meta)

    outside = tmp_path.parent / 'outside_window_0.png'
    outside.write_bytes(b'not-trusted')
    paths = _extended_paths(tmp_path, {
        'dmcount_estimates': ('window_start,zone,estimated_count,inference_ms,model_version,density_map_path\n'
                               f'0,near,6.46,10889.8,dmcount_qnrf,{outside}\n'),
    })
    with pytest.raises(InvalidExtendedResult):
        extended_storage.store_extended_result(db, seeded.id, paths)
