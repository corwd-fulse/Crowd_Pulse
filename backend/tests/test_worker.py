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


class _CrashedJob:
    """Optical Flow 중간에 프로세스가 통째로 죽은 뒤(run_once()의 예외 처리기를 전혀
    거치지 않은, recover_interrupted()가 찾아야 하는 바로 그 상태)를 실제 DB 위에
    재현한다. job은 DB에 'running'으로, checkpoint+resume_manifest는 디스크에 유효한
    상태로 남는다 — 세 가지 통합 테스트(정상 재개/중복 방지/checkpoint 손상)가 공유한다."""

    def __init__(self, db, tmp_path, monkeypatch):
        import json as json_module
        import tempfile as tempfile_module
        from pathlib import Path as PathCls

        import cv2
        import numpy as np

        from app.services import vision_results as storage_module
        from app.services.optical_flow import compute_checkpoint_fingerprint, compute_zone_motion
        from app.services.zone_analysis import load_zones_from_json
        from app.worker import TRACKED_SERVICE_FILES, sha256 as worker_sha256
        import app.worker as worker_module

        monkeypatch.setattr(storage_module, 'BASE_DIR', tmp_path)
        self.base = base = tmp_path
        (base/'uploads/videos').mkdir(parents=True)
        video_path = base/'uploads/videos/real.mp4'
        size, n, fps = 160, 30, 10
        rng = np.random.default_rng(5)
        frame0 = cv2.cvtColor(cv2.GaussianBlur(rng.integers(0, 255, (size, size), dtype=np.uint8), (15, 15), 0), cv2.COLOR_GRAY2BGR)
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (size, size))
        for i in range(n):
            writer.write(np.roll(frame0, shift=i*2, axis=1))
        writer.release()
        video_hash = worker_sha256(video_path)
        roi = {'status': 'approved', 'approved_for_analysis': True, 'video_resolution': [size, size],
              'zones': [{'name': 'A', 'polygon': [[0, 0], [size, 0], [size, size], [0, size]]}],
              'source_sha256': video_hash}
        video = Video(filename='real.mp4', saved_filename='real.mp4', storage_path='uploads/videos/real.mp4',
                     file_size=video_path.stat().st_size)
        db.add(video); db.flush()
        self.job = job = AnalysisJob(video_id=video.id, roi_snapshot=roi, analysis_profile='full')
        db.add(job); db.flush()
        model_path = base/'model.pt'
        model_path.write_bytes(b'not-a-real-model')

        def fake_track(source, out, name):
            out.mkdir(parents=True)
            det = out/'detections.csv'
            # 0-indexed frames, matching frame_count=n: a detection at frame=n-1
            # (t_ms=(n-1)*1000/fps) must stay strictly under duration=n/fps, or
            # vision_csv.validate_bundle's "offset outside video" check correctly rejects
            # the last second bucket as out-of-range.
            lines = ['frame,t_ms,track_id,x1,y1,x2,y2,conf']
            for f in range(n):
                lines.append(f'{f},{int(f*1000/fps)},1,{10+f%20},10,{20+f%20},20,0.9')
            det.write_text('\n'.join(lines)+'\n', encoding='utf-8')
            return {'detections_csv': str(det), 'frame_count': n}
        self.fake_track = fake_track

        self.sessions_factory = sessionmaker(bind=db.connection(), join_transaction_mode='create_savepoint', expire_on_commit=False)
        self.model_path = model_path
        self.db = db

        worker1 = self.make_worker()
        with worker1.exclusive():
            claim = worker1.claim_next()
            assert claim is not None
            self.claim = claim
            source, info, video_hash2 = worker1.preflight(claim)
            root = worker1.base/'uploads/analysis'/str(claim.job_id)
            root.mkdir(parents=True, exist_ok=True)
            self.root = root
            out = PathCls(tempfile_module.mkdtemp(prefix=f'{claim.token.hex}-', dir=root))
            roi_path = out/'roi_snapshot.json'
            roi_path.write_text(json_module.dumps(claim.roi), encoding='utf-8')
            summary, ai_settings = worker1.run_tracking(source, out)
            detections = PathCls(summary['detections_csv']).resolve()
            code_hashes = {p.name: worker_sha256(p) for p in (PathCls(worker_module.__file__).parent/'services').glob('*.py')
                          if p.name in TRACKED_SERVICE_FILES}
            zones = load_zones_from_json(str(roi_path))
            self.checkpoint_path = checkpoint_path = root/'flow_checkpoint.bin'
            flow_fp = compute_checkpoint_fingerprint(str(source), zones, info['fps'], worker1.frame_stride, worker1.flow_method)
            worker1._write_resume_manifest(claim, root, out, detections, video_hash2, code_hashes, 3.0,
                                           ai_settings, checkpoint_path, flow_fp)
            # "프로세스가 죽는다": run_once()를 거치지 않고 compute_zone_motion만 직접
            # 돌리다가 중간에 예외를 내서 멈춘다 — run_once()의 재시도/예외처리를 전혀
            # 거치지 않았으므로 job은 DB에 'running'으로 그대로 남는다(실제 kill -9가
            # 남기는 상태와 동일).
            def crash_partway(frame_idx, frame_count):
                if frame_idx >= 10:
                    raise RuntimeError('simulated abrupt worker death')
            with pytest.raises(RuntimeError, match='simulated abrupt worker death'):
                compute_zone_motion(str(source), zones, fps=info['fps'], frame_stride=worker1.frame_stride,
                                    flow_method=worker1.flow_method, checkpoint_path=str(checkpoint_path),
                                    checkpoint_every_frames=5, progress_callback=crash_partway)
        db.expire_all()
        assert db.get(AnalysisJob, job.id).status == 'running'
        assert checkpoint_path.exists()

    def make_worker(self, **overrides):
        kwargs = dict(base_dir=self.base, device='cpu', sessions=self.sessions_factory,
                     scope_ids=[self.job.id], track=Mock(side_effect=self.fake_track), pipeline=None,
                     frame_stride=2, heartbeat_seconds=60)
        kwargs.update(overrides)
        return Worker(self.model_path, kwargs.pop('algorithm_version', 'resume-test-v1'), **kwargs)


@pytest.mark.integration
def test_resume_after_restart_skips_tracking_and_preserves_job_identity(db,tmp_path,monkeypatch):
    """서버/프로세스가 Optical Flow 중간에 통째로 죽은 뒤 재시작한 새 Worker가, YOLO/
    ByteTrack을 다시 돌리지 않고 같은 job_id·같은 started_at으로 checkpoint부터 이어서
    끝까지 성공시키는지 확인한다. 실제 Postgres가 필요하다.
    """
    crashed = _CrashedJob(db, tmp_path, monkeypatch)
    worker2 = crashed.make_worker()
    with worker2.exclusive():
        recovered=worker2.recover_interrupted()
        assert recovered==1
        db.expire_all()
        assert db.get(AnalysisJob,crashed.job.id).status=='queued'
        assert db.get(AnalysisJob,crashed.job.id).worker_token is None
        result=worker2.run_once()
    assert result['status']=='succeeded',result
    worker2.track.assert_not_called()  # tracking 재실행 안 됨 — 이전 시도의 detections.csv 재사용
    db.expire_all()
    job_row=db.get(AnalysisJob,crashed.job.id)
    assert job_row.status=='succeeded'
    assert job_row.started_at==crashed.claim.started_at  # 연속성: 재개 전 최초 시작 시각 그대로
    assert db.get(VisionResult,crashed.job.id) is not None  # 결과가 정확히 한 번 저장됨
    assert not (crashed.root/'resume_manifest.json').exists()  # 성공 후 정리됨


@pytest.mark.integration
def test_resumed_success_stores_exactly_once_and_blocks_concurrent_worker(db,tmp_path,monkeypatch):
    """중복 DB 저장 방지: 재개로 성공한 뒤 run_once()를 또 불러도 처리할 큐가 없어 None이고,
    VisionResult는 여전히 하나뿐이다. 중복 Worker 실행 방지: 재개를 시도하는 동안(exclusive()
    보유 중) 다른 Worker 인스턴스는 전역 단일 실행 슬롯 때문에 끼어들 수 없다."""
    crashed = _CrashedJob(db, tmp_path, monkeypatch)
    from app.services import vision_results as storage_module
    store_calls=[]
    real_store=storage_module.store_result
    def counting_store(*a,**kw):
        store_calls.append(1)
        return real_store(*a,**kw)
    monkeypatch.setattr(storage_module,'store_result',counting_store)

    worker2 = crashed.make_worker()
    with worker2.exclusive():
        other = crashed.make_worker()
        with pytest.raises(WorkerBusy):
            with other.exclusive():
                pass  # 재개 작업 도중에는 두 번째 Worker가 절대 끼어들 수 없다
        assert worker2.recover_interrupted()==1
        result=worker2.run_once()
        assert result['status']=='succeeded',result
        assert worker2.run_once() is None  # 더 처리할 큐가 없음 — 재실행/재집계 없음
    assert len(store_calls)==1  # store_result()가 정확히 한 번만 호출됨
    db.expire_all()
    assert db.get(VisionResult,crashed.job.id) is not None


@pytest.mark.integration
def test_crash_before_first_flow_checkpoint_reuses_tracking(db,tmp_path,monkeypatch):
    """추적은 끝났지만 Optical Flow가 첫 checkpoint를 쓰기 전에 죽으면(파일 없음), 실패시키지 않고
    추적 결과를 재사용해 Optical Flow만 처음부터 다시 한다."""
    crashed = _CrashedJob(db, tmp_path, monkeypatch)
    crashed.checkpoint_path.unlink()

    worker2 = crashed.make_worker()
    with worker2.exclusive():
        assert worker2.recover_interrupted()==1
        db.expire_all()
        assert db.get(AnalysisJob,crashed.job.id).status=='queued'
        result=worker2.run_once()
    assert result['status']=='succeeded',result
    worker2.track.assert_not_called()
    db.expire_all()
    assert db.get(VisionResult,crashed.job.id) is not None


@pytest.mark.integration
def test_corrupted_checkpoint_after_crash_fails_with_clear_reason_not_silent_resume(db,tmp_path,monkeypatch):
    """checkpoint가 죽은 뒤 손상되면(디스크 오류 등) 임의로 재개하지 않고, 구체적인 사유와
    함께 failed로 마감한다 — WORKER_INTERRUPTED_NO_RESUME 안에 RESUME_CHECKPOINT_INVALID."""
    crashed = _CrashedJob(db, tmp_path, monkeypatch)
    crashed.checkpoint_path.write_bytes(b'corrupted, not a real checkpoint file at all')

    worker2 = crashed.make_worker()
    with worker2.exclusive():
        recovered=worker2.recover_interrupted()
        assert recovered==1
    db.expire_all()
    job_row=db.get(AnalysisJob,crashed.job.id)
    assert job_row.status=='failed'
    assert 'WORKER_INTERRUPTED_NO_RESUME' in job_row.error_message
    assert 'RESUME_CHECKPOINT_INVALID' in job_row.error_message
    worker2.track.assert_not_called()  # 재개도, 새로 돌리는 것도 전혀 시도하지 않음


# --- --auto-forecast (best-effort, never changes job status) ---------------------------

def _full(db, seeded):
    db.get(AnalysisJob, seeded.id).analysis_profile = 'full'
    db.flush()


@pytest.mark.integration
def test_auto_forecast_off_by_default_never_runs(worker, db, seeded):
    _full(db, seeded)
    runner = Mock()
    worker.forecast_runner = runner
    with worker.exclusive():
        result = worker.run_once()
    assert result['status'] == 'succeeded' and 'forecast' not in result
    runner.assert_not_called()


@pytest.mark.integration
def test_auto_forecast_reports_per_source_and_logs(worker, db, seeded):
    _full(db, seeded)
    worker.auto_forecast = True
    worker.forecast_runner = Mock(return_value={'job_id': str(seeded.id), 'sources': {'yolo': {'points': 3}, 'dmcount': {'error': 'RuntimeError: x'}}})
    with worker.exclusive():
        result = worker.run_once()
    assert result['status'] == 'succeeded'
    assert result['forecast'] == {'yolo': 'ok', 'dmcount': 'failed'}
    out = Path(result['output_dir'])
    assert 'FORECAST {' in (out / 'worker.log').read_text(encoding='utf-8')
    _, job_id, out_dir = worker.forecast_runner.call_args.args
    assert job_id == seeded.id and Path(out_dir) == out


@pytest.mark.integration
def test_auto_forecast_failure_keeps_job_succeeded(worker, db, seeded):
    _full(db, seeded)
    worker.auto_forecast = True
    worker.forecast_runner = Mock(side_effect=RuntimeError('boom'))
    with worker.exclusive():
        result = worker.run_once()
    assert result['status'] == 'succeeded' and result['forecast'] == {'error': 'RuntimeError'}
    db.expire_all()
    assert db.get(AnalysisJob, seeded.id).status == 'succeeded'
    assert 'FORECAST_FAILED (RuntimeError): boom' in (Path(result['output_dir']) / 'worker.log').read_text(encoding='utf-8')
