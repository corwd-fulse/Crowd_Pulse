import json
from unittest.mock import Mock

import cv2
import numpy as np
import pandas as pd
import pytest

from app.services.optical_flow import compute_zone_motion
from app.services.individual_flow import compute_individual_directions
from app.services.hybrid_flow import build_hybrid_summary, circular_difference
from app.services.bottleneck import compute_bottleneck_scores
from app.services.zone_analysis import Zone
from app.services.vision_csv import SCHEMAS
from app.worker import Worker
from test_vision_results import bundle, seeded, client
from test_worker import worker


def test_crop_matches_full_with_offset_roi_and_original_schema(tmp_path):
    rng = np.random.default_rng(22)
    base = cv2.GaussianBlur(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8), (11, 11), 0)
    path = tmp_path/'sample.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 360))
    for f in range(6):
        writer.write(np.roll(base, f*2, axis=1))
    writer.release()
    zones = [Zone('offset', [(220, 180), (390, 180), (390, 310), (220, 310)])]
    full = compute_zone_motion(str(path), zones)
    crop = compute_zone_motion(str(path), zones, flow_method='roi_crop_128')
    assert list(crop.columns) == SCHEMAS['flow']
    assert full[['sec', 'zone', 'low_confidence']].equals(crop[['sec', 'zone', 'low_confidence']])
    assert np.max(np.abs((full.angle_deg-crop.angle_deg+180)%360-180)) < 2
    assert np.max(np.abs(full.magnitude_px-crop.magnitude_px)) < .1


def make_tracks(tmp_path, directions):
    rows = []
    for tid, sign in enumerate(directions, 1):
        for f in range(8):
            x = 100 + tid*20 + sign*f*2
            rows.append([f, f*40, tid, x, 10, x+10, 50, .9])
    path = tmp_path/'detections.csv'
    pd.DataFrame(rows, columns=['frame','t_ms','track_id','x1','y1','x2','y2','conf']).to_csv(path,index=False)
    return path


def summary(tmp_path, directions, flow_angle=0, magnitude=1):
    path = make_tracks(tmp_path, directions)
    individual = compute_individual_directions(str(path))
    observations = pd.DataFrame([[0,'zone',len(directions),0,0]],columns=SCHEMAS['crowd_observations'])
    flow = pd.DataFrame([[0,'zone',flow_angle,magnitude,.1,False]],columns=SCHEMAS['flow'])
    return build_hybrid_summary(individual,str(path),[Zone('zone',[(0,0),(400,0),(400,100),(0,100)])],flow,observations)['rows'][0]


def test_hybrid_counts_distinct_people_normalizes_time_and_wraps_angles(tmp_path):
    row = summary(tmp_path,[1,1,1],359)
    assert row['valid_track_count'] == 3  # not the 21 contributing observations
    assert row['direction_counts']['right'] == 3
    assert row['median_speed_px_per_second'] == pytest.approx(50)
    assert row['circular_difference_deg'] == pytest.approx(1)
    assert row['reason'] == 'agreement'
    assert circular_difference(359,1) == 2


def test_opposing_people_do_not_imply_stopped_people(tmp_path):
    row = summary(tmp_path,[1,1,-1,-1])
    assert row['reason'] == 'mixed_individual_directions'
    assert row['individual_angle_deg'] is None
    assert row['agreement'] is None
    assert row['median_speed_px_per_second'] == pytest.approx(50)
    assert row['direction_counts']['left'] == 2


def test_many_observations_from_one_id_do_not_meet_support_threshold(tmp_path):
    row = summary(tmp_path,[1])
    assert row['reason'] == 'insufficient_tracks' and row['agreement'] is None


@pytest.mark.parametrize('angle,magnitude,reason',[(45,.25,'agreement'),(45.01,.25,'direction_disagreement'),
                                                (0,.249,'weak_net_flow')])
def test_hybrid_confidence_boundaries_are_not_relaxed(tmp_path,angle,magnitude,reason):
    assert summary(tmp_path,[1,1,1],angle,magnitude)['reason']==reason


def test_gap_does_not_create_reliable_movement(tmp_path):
    path = make_tracks(tmp_path,[1])
    frame = pd.read_csv(path);frame = frame[frame.frame != 3];frame.to_csv(path,index=False)
    out = compute_individual_directions(str(path))
    assert not out.loc[out.frame==4,'reliable'].iloc[0]
    assert out.loc[out.frame==4,'speed_px'].isna().all()


def test_missing_motion_omits_window_and_its_dependent_successor():
    ts = pd.DataFrame([dict(sec=s,zone='A',count=10,motion=1.,inflow=1,outflow=1) for s in range(40)])
    baseline = compute_bottleneck_scores(ts,missing_policy='omit_unavailable_v1')
    pd.testing.assert_frame_equal(baseline,compute_bottleneck_scores(ts))
    ts.loc[ts.sec==10,'motion'] = np.nan
    scores = compute_bottleneck_scores(ts,missing_policy='omit_unavailable_v1')
    assert scores.window_start.tolist() == [0,30]


def test_cpu_cannot_accidentally_select_gpu_experiment(tmp_path):
    with pytest.raises(ValueError,match='requires GPU'):
        Worker(tmp_path/'weights','test',flow_method='roi_crop_128',allow_test=True,scope_ids=['job'])


@pytest.mark.integration
def test_crop_worker_forwards_fast_and_never_runs_flow(worker,seeded,db,monkeypatch):
    from app import worker as module
    seeded.analysis_mode='test';seeded.analysis_profile='fast';db.flush()
    worker.allow_test=True;worker.device='cuda:0';worker.flow_method='roi_crop_128'
    monkeypatch.setattr(module,'check_device',lambda device:device)
    original=worker.pipeline.side_effect
    def adapter(*args,analysis_profile,flow_method):
        assert analysis_profile=='fast' and flow_method=='roi_crop_128'
        paths=original(*args)
        from pathlib import Path
        for kind in ['flow','bottleneck_scores']:
            Path(paths[kind]).write_text(','.join(SCHEMAS[kind])+'\n',encoding='utf8')
        return paths
    worker.pipeline.side_effect=adapter
    with worker.exclusive():result=worker.run_once()
    assert result['status']=='succeeded'
    worker.pipeline.assert_called_once()


@pytest.mark.integration
def test_crop_worker_rejects_production_before_tracking(worker,seeded,db):
    worker.flow_method='roi_crop_128'
    with worker.exclusive():result=worker.run_once()
    assert result['error']=='EXPERIMENT_REQUIRES_TEST_JOB'
    worker.track.assert_not_called()


@pytest.mark.integration
def test_hybrid_metadata_survives_import_and_api_without_csv_migration(db,seeded,bundle,client):
    from app.services.vision_results import store_result
    paths,roi,meta = bundle
    hybrid_row = dict(sec=0,zone='near',valid_track_count=1,individual_angle_deg=None,
                      median_speed_px_per_second=2,direction_coherence=1,
                      direction_counts=dict(right=1,down=0,left=0,up=0),
                      circular_difference_deg=None,agreement=None,low_confidence=True,reason='insufficient_tracks')
    meta = meta.model_copy(update={'settings':{'flow_method':'roi_crop_128','hybrid':{
        'rows':[hybrid_row],'bottleneck_status':[dict(window_start=0,zone='near',status='unanalyzed')]}}})
    store_result(db,seeded.id,paths,roi,meta)
    result = client.get(f'/api/analysis/jobs/{seeded.id}/results').json()
    assert result['components']['bottleneck']=='unanalyzed'
    page = client.get(f'/api/analysis/jobs/{seeded.id}/results/zone-metrics').json()
    assert page['items'][0]['hybrid']==hybrid_row
    assert page['items'][1]['hybrid'] is None
