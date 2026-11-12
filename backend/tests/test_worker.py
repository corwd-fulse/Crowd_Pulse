from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from app import worker as module
from app.database import settings
from app.models import AnalysisJob, Video, VisionResult
from app.services import vision_results as storage
from app.worker import Worker, WorkerBusy, WorkerInterrupted
from test_vision_results import bundle as bundle, seeded as seeded, client as client


@pytest.fixture
def worker(db,seeded,bundle):
    paths,roi,meta=bundle
    approved=deepcopy(roi)
    approved.update(status='approved',approved_for_analysis=True)
    seeded.roi_snapshot=approved
    db.flush()
    model=storage.BASE_DIR/'yolo11m.pt'
    model.write_bytes(b'mock-weights-not-a-real-model')
    def track(source,out,name):
        out.mkdir(parents=True)
        det=out/'detections.csv'
        det.write_text('frame,t_ms,track_id,x1,y1,x2,y2,conf\n',encoding='utf-8')
        return {'detections_csv':str(det),'frame_count':404}
    def pipeline(source,detections,zones,out_dir):
        out=Path(out_dir); out.mkdir(parents=True)
        result={}
        for kind,path in paths.items():
            destination=out/path.name
            destination.write_bytes(path.read_bytes())
            result[kind]=str(destination)
        return result
    return Worker(model,'worker-test-v1',base_dir=storage.BASE_DIR,
                  sessions=sessionmaker(bind=db.connection(),join_transaction_mode='create_savepoint',expire_on_commit=False),
                  scope_ids=[seeded.id],track=Mock(side_effect=track),pipeline=Mock(side_effect=pipeline),
                  inspect_video=Mock(return_value={'width':1920,'height':1080,'fps':25,'frame_count':404}))


@pytest.mark.integration
def test_worker_claim_run_store_and_no_duplicate(worker,db,seeded):
    job_id=seeded.id
    with worker.exclusive():
        assert worker.recover_interrupted()==0
        result=worker.run_once()
        assert result['status']=='succeeded',result
        assert worker.run_once() is None
    db.expire_all()
    job=db.get(AnalysisJob,job_id)
    assert job.status=='succeeded' and job.started_at and job.finished_at>=job.started_at
    result=db.get(VisionResult,job_id)
    assert result.execution_snapshot['analysis_mode']=='production'
    assert result.execution_snapshot['algorithm_version']=='worker-test-v1'
    assert result.execution_snapshot['direction_quality']=='unvalidated'
    assert len(result.execution_snapshot['settings']['model_sha256'])==64
    worker.track.assert_called_once()
    worker.pipeline.assert_called_once()
    dirs=list((worker.base/'uploads/analysis'/str(job_id)).iterdir())
    assert len(dirs)==1
    assert (dirs[0]/'execution_metadata.json').is_file()


@pytest.mark.integration
def test_singleton_lock_prevents_second_worker(worker):
    with worker.exclusive():
        other=Worker(worker.model_path,'test',base_dir=worker.base)
        with pytest.raises(WorkerBusy):
            with other.exclusive(): pass


@pytest.mark.integration
def test_database_lock_prevents_other_host_slot(worker,tmp_path):
    # Different local lock path simulates a separate process-local slot.
    with worker.exclusive():
        other=Worker(worker.model_path,'test',base_dir=tmp_path/'different-host')
        with pytest.raises(WorkerBusy):
            with other.exclusive(): pass


@pytest.mark.integration
@pytest.mark.parametrize('case,expected',[
    ('path','VIDEO_PATH_NOT_ALLOWED'),('missing','VIDEO_FILE_UNAVAILABLE'),
    ('candidate','ROI_NOT_APPROVED'),('roi_missing','ROI_NOT_CONFIGURED_OR_INVALID'),
    ('resolution','ROI_RESOLUTION_MISMATCH'),('hash','ROI_VIDEO_HASH_MISMATCH'),
    ('model','LOCAL_MODEL_REQUIRED_NO_AUTO_DOWNLOAD'),
])
def test_preflight_failure_no_inference(worker,db,seeded,case,expected):
    video=db.get(Video,seeded.video_id)
    if case=='path': video.storage_path='../outside.mp4'
    if case=='missing': video.storage_path='uploads/videos/missing.mp4'
    if case=='candidate': seeded.roi_snapshot=deepcopy(seeded.roi_snapshot)|{'status':'candidate','approved_for_analysis':False}
    if case=='roi_missing': seeded.roi_snapshot=None
    if case=='resolution': worker.inspect_video.return_value['width']=1280
    if case=='hash': seeded.roi_snapshot=deepcopy(seeded.roi_snapshot)|{'source_sha256':'0'*64}
    if case=='model': worker.model_path=worker.base/'missing-model.pt'
    db.flush()
    with worker.exclusive(): result=worker.run_once()
    assert result['status']=='failed' and result['error']==expected,result
    worker.track.assert_not_called()
    db.expire_all()
    assert db.get(AnalysisJob,seeded.id).status=='failed'


@pytest.mark.integration
def test_test_mode_requires_both_job_and_worker_opt_in(worker,db,seeded):
    seeded.analysis_mode='test'
    seeded.roi_snapshot=deepcopy(seeded.roi_snapshot)|{'status':'candidate','approved_for_analysis':False}
    db.flush()
    with worker.exclusive(): assert worker.run_once() is None
    worker.allow_test=True
    with worker.exclusive(): result=worker.run_once()
    assert result['status']=='succeeded',result
    db.expire_all()
    saved=db.get(VisionResult,seeded.id)
    assert saved.roi_status=='candidate'
    assert saved.execution_snapshot['analysis_mode']=='test'


@pytest.mark.integration
@pytest.mark.parametrize('component',['track','pipeline'])
def test_ai_exception_is_safe_failed_state(worker,db,seeded,component):
    getattr(worker,component).side_effect=RuntimeError('private details should not be public')
    with worker.exclusive(): result=worker.run_once()
    assert result['status']=='failed' and 'private' not in result['error']
    db.expire_all()
    assert db.get(VisionResult,seeded.id) is None
    assert db.get(AnalysisJob,seeded.id).finished_at is not None


@pytest.mark.integration
def test_storage_rollback_then_failed_status(worker,db,seeded,monkeypatch):
    original=storage.store_result
    def fail_commit(session,*args,**kwargs):
        session.commit=Mock(side_effect=SQLAlchemyError('private storage failure'))
        return original(session,*args,**kwargs)
    monkeypatch.setattr(storage,'store_result',fail_commit)
    with worker.exclusive(): result=worker.run_once()
    assert result['status']=='failed'
    db.expire_all()
    assert db.get(VisionResult,seeded.id) is None
    assert db.get(AnalysisJob,seeded.id).status=='failed'
    assert not list((worker.base/'uploads/results'/str(seeded.id)).glob('import-*'))


@pytest.mark.integration
def test_restart_recovers_abandoned_claim_without_rerun(worker,db,seeded):
    with worker.exclusive():
        claim=worker.claim_next()
        assert claim is not None
        assert worker.claim_next() is None
    # Simulate death after the committed running transition, before compute.
    with worker.exclusive():
        assert worker.recover_interrupted()==1
        assert worker.run_once() is None
    db.expire_all()
    job=db.get(AnalysisJob,seeded.id)
    assert job.status=='failed' and 'WORKER_INTERRUPTED' in job.error_message
    assert job.started_at==claim.started_at
    worker.track.assert_not_called()


@pytest.mark.integration
def test_worker_owned_job_cannot_be_manually_imported(worker,db,seeded,bundle):
    with worker.exclusive():
        claim=worker.claim_next()
        with worker.sessions() as session:
            meta=bundle[2].model_copy(update={'analysis_started_at':claim.started_at,'analysis_finished_at':datetime.now(timezone.utc)})
            with pytest.raises(storage.ResultConflict):
                storage.store_result(session,claim.job_id,bundle[0],claim.roi,meta)
        worker.fail(claim,'TEST_FINISHED')


@pytest.mark.integration
def test_interruption_marks_failed(worker,db,seeded):
    worker.track.side_effect=WorkerInterrupted('WORKER_STOP_REQUESTED')
    with worker.exclusive():
        with pytest.raises(WorkerInterrupted): worker.run_once()
    db.expire_all()
    assert db.get(AnalysisJob,seeded.id).status=='failed'


@pytest.mark.integration
def test_roi_api_snapshot_and_test_api_opt_in(worker,db,seeded,client,bundle,monkeypatch):
    from app.models import Video
    video_id=str(seeded.video_id)
    assert client.put(f'/api/videos/{video_id}/roi',json=bundle[1]).status_code==200
    assert client.get(f'/api/videos/{video_id}/roi').json()['roi_snapshot']['approved_for_analysis'] is False
    # Finish fixture queued job so duplicate index permits a new request.
    seeded.status='failed'; db.commit()
    monkeypatch.setattr(settings,'ALLOW_TEST_ANALYSIS',False)
    assert client.post('/api/analysis/jobs',json={'video_id':video_id,'analysis_mode':'test'}).status_code==403
    monkeypatch.setattr(settings,'ALLOW_TEST_ANALYSIS',True)
    response=client.post('/api/analysis/jobs',json={'video_id':video_id,'analysis_mode':'test'})
    assert response.status_code==201,response.text
    new=db.get(AnalysisJob,UUID(response.json()['id']))
    assert new.analysis_mode=='test' and new.roi_snapshot['approved_for_analysis'] is False
    # Later video edits must not silently change already requested jobs.
    approved=deepcopy(bundle[1])|{'status':'approved','approved_for_analysis':True}
    assert client.put(f'/api/videos/{video_id}/roi',json=approved).status_code==200
    db.refresh(new)
    assert new.roi_snapshot['approved_for_analysis'] is False
    assert client.post(f'/api/videos/{video_id}/analyze').status_code==410


def test_watchdog_loss_terminates_only_its_subprocess():
    code='''
from pathlib import Path
import time
from app.worker import Worker
w=Worker(Path('not-loaded.pt'),'watchdog-test',heartbeat_seconds=0.01)
calls=[]
def verify():
    calls.append(1)
    if len(calls)>1: raise RuntimeError('simulated ownership loss')
w.verify_lock=verify
with w.fail_stop_watchdog(): time.sleep(2)
'''
    result=subprocess.run([sys.executable,'-B','-c',code],timeout=15,capture_output=True)
    assert result.returncode==75


@pytest.mark.integration
def test_test_worker_flag_never_approves_production_candidate(worker,db,seeded):
    worker.allow_test=True
    seeded.roi_snapshot=deepcopy(seeded.roi_snapshot)|{'status':'candidate','approved_for_analysis':False}
    db.flush()
    with worker.exclusive(): result=worker.run_once()
    assert result['error']=='ROI_NOT_APPROVED'
    worker.track.assert_not_called()


@pytest.mark.integration
def test_each_job_has_separate_outputs_and_preserves_previous_result(worker,db,seeded):
    original_id=seeded.id
    video_id=seeded.video_id
    roi=deepcopy(seeded.roi_snapshot)
    with worker.exclusive(): first=worker.run_once()
    db.expire_all()
    previous=deepcopy(db.get(VisionResult,original_id).artifacts)
    another=AnalysisJob(video_id=video_id,roi_snapshot=roi)
    db.add(another); db.flush()
    worker.scope_ids=[another.id]
    with worker.exclusive(): second=worker.run_once()
    assert first['status']==second['status']=='succeeded'
    assert first['output_dir']!=second['output_dir']
    db.expire_all()
    assert db.get(VisionResult,original_id).artifacts==previous
    assert worker.track.call_count==2


@pytest.mark.integration
def test_fenced_old_worker_cannot_complete(worker,db,seeded,bundle):
    with worker.exclusive():
        claim=worker.claim_next()
        with worker.sessions() as session:
            job=session.get(AnalysisJob,claim.job_id)
            job.worker_token=uuid4(); session.commit()
        with worker.sessions() as session:
            meta=bundle[2].model_copy(update={'analysis_started_at':claim.started_at,'analysis_finished_at':datetime.now(timezone.utc)})
            with pytest.raises(storage.ResultConflict):
                storage.store_result(session,claim.job_id,bundle[0],claim.roi,meta,worker_token=claim.token)
        worker.fail(claim,'STALE_WORKER_MUST_NOT_CHANGE_STATUS')
    db.expire_all()
    assert db.get(AnalysisJob,claim.job_id).status=='running'


@pytest.mark.integration
def test_pipeline_cannot_return_external_paths(worker,db,seeded,bundle):
    worker.pipeline.side_effect=lambda *args: {kind:str(path) for kind,path in bundle[0].items()}
    with worker.exclusive(): result=worker.run_once()
    assert result['error']=='PIPELINE_OUTPUT_PATH_NOT_ALLOWED'
    db.expire_all()
    assert db.get(VisionResult,seeded.id) is None
