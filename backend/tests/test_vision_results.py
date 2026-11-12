from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from pydantic import ValidationError

from app.database import get_db
from app.main import app
from app.models import AnalysisJob, Video, VisionResult, VisionZoneMetric, VisionBottleneckScore
from app.services import vision_results as service
from app.services.vision_csv import ExecutionMetadata, InvalidResult, SCHEMAS, validate_bundle

SAMPLES = Path(__file__).resolve().parents[1] / 'samples'


@pytest.fixture
def bundle(tmp_path):
    roi = {'status':'candidate_unvalidated', 'approved_for_analysis':False,
           'video_resolution':[1920,1080], 'zones':[{'name':'near','polygon':[[0,0],[50,0],[50,50],[0,50]]}]}
    started = datetime(2026, 11, 12, tzinfo=timezone.utc)
    meta = ExecutionMetadata(analysis_started_at=started, analysis_finished_at=started+timedelta(minutes=18),
                             model_version='yolo11m-fixture', algorithm_version='vision-fixture-v1', video_sha256=hashlib.sha256(b'fixture-video').hexdigest(), fps=25, frame_count=404)
    contents = {
        'crowd_observations':'sec,zone,count,inflow,outflow\n0,near,3.9,1,2\n1,near,1.7,0,1\n',
        'flow':'sec,zone,angle_deg,magnitude_px,motion,low_confidence\n0,near,226.65,0.9,0.04,False\n1,near,,,,True\n',
        'individual_directions':'frame,t_ms,track_id,angle_deg,speed_px,reliable\n1,40,1,203.14,0.25,True\n2,80,1,,,False\n3,120,1,,0,True\n',
        'individual_zone_agg':'sec,zone,mean_angle_deg,n_tracks\n0,near,203.14,16\n',
        'bottleneck_scores':'window_start,zone,score,growth,slowdown,imbalance,risk_level\n0,near,5.9,0.1,0.01,0.02,낮음\n10,near,45,0.4,0.5,0.2,주의\n',
    }
    paths = {}
    for name, content in contents.items():
        paths[name] = tmp_path / f'{name}.csv'
        paths[name].write_text(content, encoding='utf-8')
    return paths, roi, meta


@pytest.fixture
def seeded(db, tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'BASE_DIR', tmp_path)
    folder = tmp_path / 'uploads/videos'
    folder.mkdir(parents=True)
    uid=uuid4()
    name=f'{uid.hex}.mp4'
    (folder/name).write_bytes(b'fixture-video')
    video=Video(id=uid, filename='fixture.mp4', saved_filename=name, storage_path=f'uploads/videos/{name}',file_size=13)
    db.add(video)
    db.flush()
    job=AnalysisJob(video_id=uid)
    db.add(job)
    db.flush()
    return job


@pytest.fixture
def client(db):
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def upload(client, job_id, bundle):
    paths,roi,meta=bundle
    return client.post(f'/api/analysis/jobs/{job_id}/results', data={
        'roi_snapshot':json.dumps(roi), 'execution_metadata':meta.model_dump_json(),
    },files={kind:(f'{kind}.csv', path.read_bytes(),'text/csv') for kind,path in paths.items()})


def test_validation_preserves_nullable_raw_rows_and_decimal_count(bundle):
    paths,roi,meta=bundle
    status,manifest,records=validate_bundle(paths,roi,meta)
    assert status=='candidate'
    assert records['crowd_observations'][0]['count']==3.9
    assert manifest['individual_directions']['rows']==3
    assert manifest['individual_directions']['missing']['angle_deg']==2
    assert records['flow'][1]['angle_deg'] is None


@pytest.mark.parametrize('kind,before,after',[
    ('crowd_observations','3.9','NaN'),
    ('crowd_observations','3.9','-1'),
    ('crowd_observations','3.9','3.123456'),
    ('crowd_observations','near','unknown'),
    ('flow','226.65','360'),
    ('flow','False','0'),
    ('individual_directions','2,80,1,,,False','2,80,1,90,1,False'),
    ('individual_directions','1,40','1,41'),
    ('individual_zone_agg','203.14','Infinity'),
    ('bottleneck_scores','5.9','101'),
    ('bottleneck_scores','낮음','critical'),
])
def test_invalid_csv_rejected(bundle,kind,before,after):
    paths,roi,meta=bundle
    paths[kind].write_text(paths[kind].read_text(encoding='utf-8').replace(before,after),encoding='utf-8')
    with pytest.raises(InvalidResult): validate_bundle(paths,roi,meta)


def test_duplicate_csv_row_and_invalid_roi_rejected(bundle):
    paths,roi,meta=bundle
    path=paths['crowd_observations']
    original=path.read_text(encoding='utf-8')
    path.write_text(original+original.splitlines()[1]+'\n',encoding='utf-8')
    with pytest.raises(InvalidResult): validate_bundle(paths,roi,meta)
    path.write_text(original,encoding='utf-8')
    roi['zones'][0]['polygon'][0]=[1921,0]
    with pytest.raises(InvalidResult): validate_bundle(paths,roi,meta)


def test_repository_sample_csv_validation(bundle):
    _,roi,meta=bundle
    source=json.loads((SAMPLES.parent/'configs/zones/alley_entrance_inner.json').read_text(encoding='utf-8'))
    source.update(status='candidate',approved_for_analysis=False)
    meta=meta.model_copy(update={'fps':30.0,'frame_count':1900})
    paths={name:SAMPLES/f'{name}.csv' for name in SCHEMAS}
    _,manifest,_=validate_bundle(paths,source,meta)
    assert all(entry['rows']>0 for entry in manifest.values())


@pytest.mark.integration
def test_save_query_download_idempotency_and_conflict(db,seeded,client,bundle):
    first=upload(client,seeded.id,bundle)
    assert first.status_code==201,first.text
    assert first.json()['roi_status']=='candidate'
    result=db.get(VisionResult,seeded.id)
    row=db.scalar(select(VisionZoneMetric).where(VisionZoneMetric.analysis_job_id==seeded.id,VisionZoneMetric.video_offset_ms==0))
    assert row.mean_person_count==Decimal('3.9000')
    assert row.direction_sample_count==16
    later=db.scalar(select(VisionZoneMetric).where(VisionZoneMetric.analysis_job_id==seeded.id,VisionZoneMetric.video_offset_ms==1000))
    assert later.low_confidence and later.motion is None
    assert not later.has_individual_aggregate and later.direction_sample_count is None
    db.refresh(seeded)
    assert seeded.status=='succeeded'
    assert seeded.started_at==bundle[2].analysis_started_at
    assert result.imported_at!=seeded.finished_at
    base=f'/api/analysis/jobs/{seeded.id}/results'
    summary=client.get(base).json()
    assert summary['video_id']==str(seeded.video_id)
    assert 'path' not in summary['artifacts']['individual_directions']
    assert summary['artifacts']['individual_directions']['missing']['angle_deg']==2
    assert client.get(base+'/artifacts/individual_directions').content==bundle[0]['individual_directions'].read_bytes()
    assert client.get(base+'/zone-metrics?region_key=near&limit=1').json()['items'][0]['mean_person_count']==3.9
    assert len(client.get(base+'/zone-metrics?start_ms=1000').json()['items'])==1
    scores=client.get(base+'/bottleneck-scores').json()['items']
    assert [r['severity'] for r in scores]==['low','medium']
    assert upload(client,seeded.id,bundle).status_code==200
    assert db.scalar(select(func.count()).select_from(VisionZoneMetric).where(VisionZoneMetric.analysis_job_id==seeded.id))==2
    path=bundle[0]['crowd_observations']
    path.write_text(path.read_text(encoding='utf-8').replace('3.9','4.1'),encoding='utf-8')
    assert upload(client,seeded.id,bundle).status_code==409
    assert client.get(f'/api/analysis/jobs/{seeded.id}').status_code==200
    assert client.get(f'/api/analysis/jobs?video_id={seeded.video_id}').status_code==200


@pytest.mark.integration
def test_empty_bottleneck_is_success(db,seeded,bundle):
    paths,roi,meta=bundle
    paths['bottleneck_scores'].write_text(','.join(SCHEMAS['bottleneck_scores'])+'\n',encoding='utf-8')
    result,created=service.store_result(db,seeded.id,paths,roi,meta)
    assert created and result.artifacts['bottleneck_scores']['rows']==0
    assert db.scalar(select(func.count()).select_from(VisionBottleneckScore).where(VisionBottleneckScore.analysis_job_id==seeded.id))==0
    assert db.get(AnalysisJob,seeded.id).status=='succeeded'


@pytest.mark.integration
def test_failure_rolls_back_database_and_new_files(db,seeded,bundle,monkeypatch):
    job_id=seeded.id
    # Persist only the fixture within the outer rollback transaction first.
    db.commit()
    original=db.execute
    def fail_on_scores(statement,*args,**kwargs):
        if getattr(getattr(statement,'table',None),'name',None)=='vision_bottleneck_scores':
            raise SQLAlchemyError('simulated storage failure')
        return original(statement,*args,**kwargs)
    monkeypatch.setattr(db,'execute',fail_on_scores)
    with pytest.raises(SQLAlchemyError): service.store_result(db,job_id,*bundle)
    monkeypatch.setattr(db,'execute',original)
    assert db.get(VisionResult,job_id) is None
    assert db.get(AnalysisJob,job_id).status=='queued'
    assert db.scalar(select(func.count()).select_from(VisionZoneMetric).where(VisionZoneMetric.analysis_job_id==job_id))==0
    assert not list((service.BASE_DIR/'uploads/results'/str(job_id)).glob('import-*'))


@pytest.mark.integration
def test_bad_request_no_partial_data_and_missing_result(db,seeded,client,bundle):
    assert client.get(f'/api/analysis/jobs/{seeded.id}/results').status_code==404
    bundle[1]['approved_for_analysis']=True
    assert upload(client,seeded.id,bundle).status_code==422
    assert db.get(VisionResult,seeded.id) is None
    assert client.get(f'/api/analysis/jobs/{uuid4()}/results').status_code==404


@pytest.mark.integration
def test_source_video_hash_mismatch(db,seeded,bundle):
    paths,roi,meta=bundle
    meta=meta.model_copy(update={'video_sha256':'0'*64})
    job_id=seeded.id
    db.commit()
    with pytest.raises(InvalidResult): service.store_result(db,job_id,paths,roi,meta)
    assert db.get(VisionResult,job_id) is None


@pytest.mark.integration
def test_new_hypertables_and_partition_keys(db):
    tables=set(db.execute(text("SELECT hypertable_name FROM timescaledb_information.hypertables WHERE hypertable_schema='public'")).scalars())
    assert {'vision_zone_metrics','vision_bottleneck_scores'}<=tables
    for model in [VisionZoneMetric,VisionBottleneckScore]:
        assert not model.__table__.c.imported_at.nullable
        assert 'imported_at' in model.__table__.primary_key.columns


@pytest.mark.integration
def test_repository_samples_store_and_read(db,seeded,client,bundle):
    # Metadata/source video here belong to the isolated fixture, not production.
    _,_,meta=bundle
    roi=json.loads((SAMPLES.parent/'configs/zones/alley_entrance_inner.json').read_text(encoding='utf-8'))
    roi.update(status='candidate',approved_for_analysis=False)
    meta=meta.model_copy(update={'fps':30.0,'frame_count':1900})
    paths={name:SAMPLES/f'{name}.csv' for name in SCHEMAS}
    response=upload(client,seeded.id,(paths,roi,meta))
    assert response.status_code==201,response.text
    base=f'/api/analysis/jobs/{seeded.id}/results'
    summary=client.get(base).json()
    _,manifest,_=validate_bundle(paths,roi,meta)
    for kind in SCHEMAS:
        assert summary['artifacts'][kind]['rows']==manifest[kind]['rows']
        assert client.get(base+'/artifacts/'+kind).content==paths[kind].read_bytes()
    scores=client.get(base+'/bottleneck-scores?limit=1000').json()['items']
    assert len(scores)==manifest['bottleneck_scores']['rows']


@pytest.mark.integration
def test_approved_snapshot_and_high_risk_mapping(db,seeded,bundle):
    paths,roi,meta=bundle
    roi.update(status='approved',approved_for_analysis=True)
    path=paths['bottleneck_scores']
    path.write_text(path.read_text(encoding='utf-8').replace('45,0.4,0.5,0.2,주의','75,0.4,0.5,0.2,높음'),encoding='utf-8')
    result,_=service.store_result(db,seeded.id,paths,roi,meta)
    assert result.roi_status=='approved'
    high=db.scalar(select(VisionBottleneckScore).where(VisionBottleneckScore.analysis_job_id==seeded.id,VisionBottleneckScore.risk_level=='높음'))
    assert high.severity=='high'


@pytest.mark.integration
def test_commit_failure_cleans_files_and_returns_safe_error(db,seeded,client,bundle,monkeypatch):
    job_id=seeded.id
    db.commit()
    def fail_commit():
        raise SQLAlchemyError('private commit details')
    monkeypatch.setattr(db,'commit',fail_commit)
    response=upload(client,job_id,bundle)
    assert response.status_code==503
    assert 'private' not in response.text
    assert db.get(VisionResult,job_id) is None
    assert not list((service.BASE_DIR/'uploads/results'/str(job_id)).glob('import-*'))


@pytest.mark.integration
def test_failed_job_import_rejected(db,seeded,client,bundle):
    seeded.status='failed'
    db.commit()
    response=upload(client,seeded.id,bundle)
    assert response.status_code==409


@pytest.mark.integration
def test_missing_job_invalid_ranges_and_artifact_kind(client):
    assert client.get(f'/api/analysis/jobs/{uuid4()}/results/zone-metrics').status_code==404
    assert client.get(f'/api/analysis/jobs/{uuid4()}/results/zone-metrics?limit=1001').status_code==422


def test_actual_execution_times_require_timezone_and_order(bundle):
    meta=bundle[2].model_dump()
    meta['analysis_started_at']=datetime(2026,11,12)
    with pytest.raises(ValidationError): ExecutionMetadata.model_validate(meta)
    meta=bundle[2].model_dump()
    meta['analysis_finished_at']=meta['analysis_started_at']-timedelta(seconds=1)
    with pytest.raises(ValidationError): ExecutionMetadata.model_validate(meta)


@pytest.mark.integration
def test_empty_individual_and_aggregate_files_are_preserved(db,seeded,bundle):
    paths,roi,meta=bundle
    for kind in ['individual_directions','individual_zone_agg','bottleneck_scores']:
        paths[kind].write_text(','.join(SCHEMAS[kind])+'\n',encoding='utf-8')
    result,_=service.store_result(db,seeded.id,paths,roi,meta)
    assert result.artifacts['individual_directions']['rows']==0
    rows=db.scalars(select(VisionZoneMetric).where(VisionZoneMetric.analysis_job_id==seeded.id)).all()
    assert rows and all(not r.has_individual_aggregate and r.mean_angle_deg is None for r in rows)


@pytest.mark.integration
def test_existing_job_creation_and_duplicate_guard_unchanged(db,seeded,client,bundle):
    video_id=str(seeded.video_id)
    assert client.post('/api/analysis/jobs',json={'video_id':video_id}).status_code==409
    service.store_result(db,seeded.id,*bundle)
    response=client.post('/api/analysis/jobs',json={'video_id':video_id})
    assert response.status_code==201,response.text
    assert response.json()['status']=='queued'
    assert client.post('/api/analysis/jobs',json={'video_id':video_id}).status_code==409
    assert client.get('/api/analysis/jobs/'+response.json()['id']).status_code==200


@pytest.mark.integration
def test_result_read_validation_and_artifact_whitelist(db,seeded,client,bundle):
    service.store_result(db,seeded.id,*bundle)
    base=f'/api/analysis/jobs/{seeded.id}/results'
    assert client.get(base+'/zone-metrics?start_ms=1000&end_ms=0').status_code==422
    assert client.get(base+'/artifacts/unknown').status_code==404
    assert client.get(base+'/zone-metrics?region_key=absent').json()['items']==[]


@pytest.mark.integration
def test_direction_defaults_unvalidated_and_versions_preserved(db,seeded,client,bundle):
    paths,roi,meta=bundle
    service.store_result(db,seeded.id,paths,roi,meta)
    base=f'/api/analysis/jobs/{seeded.id}/results'
    summary=client.get(base).json()
    assert summary['execution_snapshot']['algorithm_version']=='vision-fixture-v1'
    assert summary['direction_presentation']['quality']=='unvalidated'
    assert not summary['direction_presentation']['confirmed_for_display']
    for row in client.get(base+'/zone-metrics').json()['items']:
        assert not row['flow_direction_confirmed'] and not row['individual_direction_confirmed']


def test_algorithm_version_required_and_validation_evidence_required(bundle):
    data=bundle[2].model_dump()
    del data['algorithm_version']
    with pytest.raises(ValidationError): ExecutionMetadata.model_validate(data)
    data=bundle[2].model_dump() | {'direction_quality':'validated'}
    with pytest.raises(ValidationError): ExecutionMetadata.model_validate(data)


@pytest.mark.integration
def test_reviewed_result_still_preserves_low_confidence_and_absent_direction(db,seeded,client,bundle):
    paths,roi,meta=bundle
    roi.update(status='approved',approved_for_analysis=True)
    meta=ExecutionMetadata.model_validate(meta.model_dump() | {
        'algorithm_version':'flow-circular-v2-test', 'direction_quality':'validated',
        'direction_review':{'reviewer':'test reviewer','evidence':'fixture only; not production approval'},
    })
    service.store_result(db,seeded.id,paths,roi,meta)
    rows=client.get(f'/api/analysis/jobs/{seeded.id}/results/zone-metrics').json()['items']
    assert rows[0]['flow_direction_confirmed'] and rows[0]['individual_direction_confirmed']
    assert rows[1]['low_confidence'] is True
    assert not rows[1]['flow_direction_confirmed'] and not rows[1]['individual_direction_confirmed']


@pytest.mark.integration
def test_future_algorithm_window_does_not_require_schema_change(db,seeded,bundle):
    paths,roi,meta=bundle
    path=paths['bottleneck_scores']
    path.write_text(path.read_text(encoding='utf-8').replace('10,near','7,near'),encoding='utf-8')
    service.store_result(db,seeded.id,paths,roi,meta)
    assert db.scalar(select(VisionBottleneckScore).where(VisionBottleneckScore.analysis_job_id==seeded.id,VisionBottleneckScore.window_start_ms==7000)) is not None


@pytest.mark.integration
def test_algorithm_reanalysis_uses_new_job_and_keeps_old_snapshot(db,seeded,bundle):
    paths,roi,meta=bundle
    old_id=seeded.id
    video_id=seeded.video_id
    service.store_result(db,old_id,paths,roi,meta)
    new_job=AnalysisJob(video_id=video_id)
    db.add(new_job); db.flush()
    new_meta=ExecutionMetadata.model_validate(meta.model_dump() | {'algorithm_version':'vision-fixture-v2','settings':{'angle_aggregation':'candidate-new-method'}})
    service.store_result(db,new_job.id,paths,roi,new_meta)
    assert db.get(VisionResult,old_id).execution_snapshot['algorithm_version']=='vision-fixture-v1'
    assert db.get(VisionResult,new_job.id).execution_snapshot['algorithm_version']=='vision-fixture-v2'
