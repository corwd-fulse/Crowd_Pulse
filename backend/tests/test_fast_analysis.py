from pathlib import Path
import hashlib
import json
from unittest.mock import Mock

import pandas as pd
import pytest
from sqlalchemy import select

from app.models import AnalysisJob, VisionZoneMetric, VisionResult
from app.services import pipeline, vision_results
from app.services.vision_csv import SCHEMAS, InvalidResult, validate_bundle
from test_vision_results import bundle as bundle, seeded as seeded, client as client
from test_worker import worker as worker


@pytest.mark.parametrize('profile', ['fast','full'])
def test_pipeline_profile_selects_only_requested_components(tmp_path, monkeypatch, profile):
    cap=Mock(); cap.get.return_value=25
    monkeypatch.setattr(pipeline.cv2,'VideoCapture',Mock(return_value=cap))
    functions={
        'compute_zone_timeseries':pd.DataFrame([[0,'near',2.5,1,0]],columns=SCHEMAS['crowd_observations']),
        'compute_zone_motion':pd.DataFrame([[0,'near',30,1,.1,False]],columns=SCHEMAS['flow']),
        'compute_individual_directions':pd.DataFrame(columns=SCHEMAS['individual_directions']),
        'aggregate_by_zone':pd.DataFrame(columns=SCHEMAS['individual_zone_agg']),
        'compute_bottleneck_scores':pd.DataFrame(columns=SCHEMAS['bottleneck_scores']),
    }
    mocks={name:Mock(return_value=df) for name,df in functions.items()}
    for name,m in mocks.items(): monkeypatch.setattr(pipeline,name,m)
    paths=pipeline.run_zone_pipeline('video','detections',[],str(tmp_path),analysis_profile=profile)
    assert set(paths)==set(SCHEMAS)
    for key,path in paths.items():assert list(pd.read_csv(path).columns)==SCHEMAS[key]
    cap.release.assert_called_once()
    for key in ['compute_zone_motion','compute_bottleneck_scores']:
        assert mocks[key].call_count==(1 if profile=='full' else 0)
    for key in ['compute_zone_timeseries','compute_individual_directions','aggregate_by_zone']:
        mocks[key].assert_called_once()


def make_fast(bundle):
    paths,roi,meta=bundle
    for kind in ('flow','bottleneck_scores'):
        paths[kind].write_text(','.join(SCHEMAS[kind])+'\n',encoding='utf8')
    return paths,roi,meta.model_copy(update={'analysis_profile':'fast'})


def test_fast_rejects_claimed_optical_results(bundle):
    paths,roi,meta=bundle
    with pytest.raises(InvalidResult,match='FAST'):
        validate_bundle(paths,roi,meta.model_copy(update={'analysis_profile':'fast'}))


@pytest.mark.integration
def test_full_import_keeps_legacy_fingerprint(db,seeded,bundle):
    paths,roi,meta=bundle
    _,manifest,_=validate_bundle(paths,roi,meta)
    old_metadata=meta.model_dump(mode='json');old_metadata.pop('analysis_profile')
    expected=hashlib.sha256(json.dumps({'roi':roi,'execution':old_metadata,'files':manifest},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    result,_=vision_results.store_result(db,seeded.id,paths,roi,meta)
    assert result.fingerprint==expected


@pytest.mark.integration
def test_fast_storage_query_idempotency_and_missing_components(db,seeded,client,bundle):
    paths,roi,meta=make_fast(bundle)
    seeded.analysis_profile='fast';db.flush()
    saved,created=vision_results.store_result(db,seeded.id,paths,roi,meta)
    assert created
    assert vision_results.store_result(db,seeded.id,paths,roi,meta)[1] is False
    row=db.scalar(select(VisionZoneMetric).where(VisionZoneMetric.analysis_job_id==seeded.id))
    assert row.has_observation and not row.has_flow
    assert row.low_confidence is None and row.motion is None
    summary=client.get(f'/api/analysis/jobs/{seeded.id}/results').json()
    assert summary['analysis_profile']=='fast'
    assert summary['components']['optical_flow']=='not_run'
    assert summary['components']['bottleneck']=='not_run'
    assert summary['artifacts']['flow']['rows']==0
    score=client.get(f'/api/analysis/jobs/{seeded.id}/results/bottleneck-scores').json()
    assert score['items']==[] and score['component_status']=='not_run'
    assert client.get(f'/api/analysis/jobs/{seeded.id}').json()['status']=='succeeded'


@pytest.mark.integration
def test_profile_mismatch_cannot_save(db,seeded,bundle):
    paths,roi,meta=make_fast(bundle)
    with pytest.raises(vision_results.ResultConflict,match='profile'):
        vision_results.store_result(db,seeded.id,paths,roi,meta)
    assert db.get(VisionResult,seeded.id) is None


@pytest.mark.integration
def test_request_defaults_full_and_duplicate_policy_shared_across_profiles(db,seeded,client):
    assert seeded.analysis_profile=='full'
    assert client.post('/api/analysis/jobs',json={'video_id':str(seeded.video_id),'analysis_profile':'fast'}).status_code==409
    seeded.status='cancelled';db.flush()
    response=client.post('/api/analysis/jobs',json={'video_id':str(seeded.video_id),'analysis_profile':'fast'})
    assert response.status_code==201,response.text
    assert response.json()['analysis_profile']=='fast'
    assert client.post('/api/analysis/jobs',json={'video_id':str(seeded.video_id)}).status_code==409
    assert client.post('/api/analysis/jobs',json={'video_id':str(seeded.video_id),'analysis_profile':'invalid'}).status_code==422


@pytest.mark.integration
def test_worker_passes_fast_profile_and_stores_result(worker,seeded,db):
    seeded.analysis_profile='fast';db.flush()
    original=worker.pipeline.side_effect
    def fast_pipeline(*args,analysis_profile):
        assert analysis_profile=='fast'
        paths=original(*args)
        for kind in ('flow','bottleneck_scores'):
            Path(paths[kind]).write_text(','.join(SCHEMAS[kind])+'\n',encoding='utf8')
        return paths
    worker.pipeline.side_effect=fast_pipeline
    with worker.exclusive():result=worker.run_once()
    assert result['status']=='succeeded',result
    db.expire_all()
    stored=db.get(VisionResult,seeded.id)
    assert stored.execution_snapshot['analysis_profile']=='fast'
    assert stored.artifacts['flow']['rows']==0
    worker.track.assert_called_once()
