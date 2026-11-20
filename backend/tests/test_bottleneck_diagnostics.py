from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from app.services.bottleneck import compute_bottleneck_scores
from app.services.bottleneck_diagnostics import describe_bottleneck_windows
from app.services import optical_flow
from app.services.zone_analysis import Zone
from test_vision_results import bundle, seeded, client


def inputs(seconds=20, motion=1.):
    return pd.DataFrame([dict(sec=s,zone='A',count=5.,motion=motion,inflow=1,outflow=1,
                              low_confidence=False) for s in range(seconds)])


def diagnose(ts):
    scores=compute_bottleneck_scores(ts,missing_policy='omit_unavailable_v1')
    keys={(int(r.window_start),r.zone) for r in scores.itertuples()}
    return scores,describe_bottleneck_windows(ts.to_dict('records'),keys)


@pytest.mark.parametrize('motion',[0.,0.001,1.])
def test_valid_calm_and_stationary_inputs_produce_real_zero_not_missing(motion):
    ts=inputs(motion=motion)
    ts['hybrid']=[dict(reason='weak_net_flow',valid_track_count=3,direction_coherence=1.)]*len(ts)
    scores,status=diagnose(ts)
    assert scores.score.tolist()==[0.,0.]
    assert scores.risk_level.tolist()==['낮음','낮음']
    assert all(r['status']=='analyzed' and r['hybrid_affects_bottleneck'] is False for r in status)


def test_complete_congestion_inputs_can_raise_high_risk():
    ts=inputs()
    ts.loc[ts.sec<10,['count','motion']]=[1.,10.]
    ts.loc[ts.sec>=10,['count','motion','inflow','outflow']]=[10.,0.,10,0]
    scores,status=diagnose(ts)
    assert scores.iloc[1].score==96.
    assert scores.iloc[1].risk_level=='높음'
    assert status[1]['status']=='analyzed'


def test_each_missing_metric_and_previous_window_are_explained():
    ts=inputs();ts.loc[ts.sec==4,'motion']=np.nan;ts.loc[ts.sec==4,'low_confidence']=True
    ts.loc[ts.sec==11,'inflow']=np.nan
    scores,status=diagnose(ts)
    assert scores.empty
    assert status[0]['missing_metrics']['motion']==[4]
    assert status[0]['flow_low_confidence_seconds']==[4]
    assert 'missing_inflow' in status[1]['reasons']
    assert 'previous_window_missing_metrics' in status[1]['reasons']
    assert status[1]['previous_missing_metrics']['motion']==[4]


def test_insufficient_windows_differs_from_weak_motion_and_disagreement():
    ts=inputs(seconds=4)
    ts['hybrid']=[dict(reason=reason,valid_track_count=3,direction_coherence=1.) for reason in
                  ['insufficient_tracks','weak_net_flow','direction_disagreement','mixed_individual_directions']]
    scores,status=diagnose(ts)
    assert scores.empty and status[0]['reasons']==['insufficient_windows']
    assert status[0]['hybrid']['issue_seconds']['weak_net_flow']==[1]
    assert status[0]['hybrid']['issue_seconds']['direction_disagreement']==[2]


@pytest.mark.parametrize('brightness,side,expected',[(40,4,False),(41,4,True),(0,3,True)])
@pytest.mark.parametrize('flow_method',['full_frame','roi_crop_128'])
def test_actual_flow_confidence_threshold_and_mask_support(monkeypatch,brightness,side,expected,flow_method):
    cap=Mock()
    cap.get.side_effect=lambda key:{optical_flow.cv2.CAP_PROP_FPS:25,
        optical_flow.cv2.CAP_PROP_FRAME_WIDTH:16,optical_flow.cv2.CAP_PROP_FRAME_HEIGHT:16}[key]
    cap.read.side_effect=[(True,np.zeros((16,16,3),np.uint8)),
                          (True,np.full((16,16,3),brightness,np.uint8)),(False,None)]
    monkeypatch.setattr(optical_flow.cv2,'VideoCapture',Mock(return_value=cap))
    monkeypatch.setattr(optical_flow.cv2,'calcOpticalFlowFarneback',Mock(return_value=np.zeros((16,16,2),np.float32)))
    result=optical_flow.compute_zone_motion('unused',[Zone('A',[(0,0),(side,0),(side,side),(0,side)])],flow_method=flow_method)
    assert bool(result.iloc[0].low_confidence)==expected
    assert bool(result.iloc[0:1].motion.isna().all())==expected


@pytest.mark.integration
def test_api_derives_detail_without_rewriting_old_snapshot(db,seeded,bundle,client):
    from app.services.vision_results import store_result
    from app.models import VisionResult
    paths,roi,meta=bundle
    paths['bottleneck_scores'].write_text('window_start,zone,score,growth,slowdown,imbalance,risk_level\n',encoding='utf8')
    old_status=[dict(window_start=0,zone='near',status='unanalyzed',reason='insufficient_windows_or_missing_metrics')]
    meta=meta.model_copy(update={'settings':{'hybrid':{'bottleneck_status':old_status}}})
    saved,_=store_result(db,seeded.id,paths,roi,meta)
    snapshot=saved.execution_snapshot.copy()
    response=client.get(f'/api/analysis/jobs/{seeded.id}/results')
    assert response.status_code==200
    detail=response.json()['bottleneck_status'][0]
    assert detail['missing_metrics']['motion']==[1]
    assert detail['flow_low_confidence_seconds']==[1]
    assert detail['status']=='unanalyzed'
    db.expire_all()
    assert db.get(VisionResult,seeded.id).execution_snapshot==snapshot
