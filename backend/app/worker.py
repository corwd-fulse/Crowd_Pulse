"""Independent, single-host Vision worker. Run explicitly; never on API startup."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
from uuid import UUID, uuid4

from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

from app.database import engine
from app.models import AnalysisJob, Video, VisionResult
from app.services import vision_results as storage
from app.services.vision_csv import ExecutionMetadata, InvalidResult, SCHEMAS, validate_roi
from app.services.inference_runtime import normalize_device, check_device, InferenceFailure, is_cuda_oom

LOCK_KEY = 19201112


class WorkerBusy(RuntimeError):
    pass


class WorkerInterrupted(Exception):
    pass


class InputFailure(ValueError):
    pass


@dataclass(frozen=True)
class Claim:
    job_id: UUID
    token: UUID
    started_at: datetime
    analysis_mode: str
    storage_path: str
    roi: dict | None
    video_status: str
    analysis_profile: str = 'full'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def video_info(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        ok, _ = cap.read()
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not ok or not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise InputFailure('VIDEO_METADATA_UNAVAILABLE')
        return {'width':int(cap.get(3)), 'height':int(cap.get(4)), 'fps':fps, 'frame_count':count}
    finally:
        cap.release()


class Worker:
    def __init__(self, model_path: Path, algorithm_version: str, *, allow_test=False,
                 bind=engine, sessions=None, base_dir=None, track=None, pipeline=None,
                 inspect_video=video_info, heartbeat_seconds=2.0, scope_ids=None, device='cpu', imgsz=1920):
        self.bind = bind
        self.sessions = sessions or sessionmaker(bind, expire_on_commit=False)
        self.base = Path(base_dir or storage.BASE_DIR).resolve()
        self.model_path = Path(model_path).resolve()
        self.device = normalize_device(device)
        if type(imgsz) is not int or imgsz <= 0 or imgsz % 32:
            raise ValueError('imgsz must be a positive multiple of 32')
        self.imgsz = imgsz
        if not algorithm_version.strip():
            raise ValueError('Explicit algorithm version is required')
        self.algorithm_version = algorithm_version
        self.allow_test = allow_test
        self.track = track
        self.pipeline = pipeline
        self.inspect_video = inspect_video
        self.heartbeat_seconds = heartbeat_seconds
        # Isolated tests and explicit --once --job-id checks restrict queue scope.
        self.scope_ids = scope_ids
        self._control = None
        self._lost = threading.Event()

    def scoped(self, query):
        return query.where(AnalysisJob.id.in_(self.scope_ids)) if self.scope_ids is not None else query

    @contextmanager
    def exclusive(self):
        # Local lock remains held until this process exits, even if the DB session
        # disconnects. It prevents another worker on this host entering YOLO.
        lock_root = self.base / 'uploads'
        lock_root.mkdir(parents=True, exist_ok=True)
        handle = (lock_root / '.vision-worker.lock').open('a+b')
        locked = False
        try:
            if handle.tell() == 0:
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise WorkerBusy('A Vision worker already owns the local inference slot') from None
            locked = True
            with self.bind.connect() as control:
                if not control.execute(text('SELECT pg_try_advisory_lock(:key)'), {'key':LOCK_KEY}).scalar_one():
                    control.rollback()
                    raise WorkerBusy('A Vision worker already owns the database inference slot')
                control.commit()
                self._control = control
                try:
                    yield self
                finally:
                    self._control = None
                    if not control.invalidated:
                        try:
                            control.execute(text('SELECT pg_advisory_unlock(:key)'), {'key':LOCK_KEY})
                            control.commit()
                        except Exception:
                            control.rollback()
        finally:
            if locked:
                handle.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def verify_lock(self):
        if self._control is None or self._lost.is_set() or self._control.invalidated:
            raise WorkerInterrupted('WORKER_LOCK_LOST')
        # Never silently reconnect and continue computation without ownership.
        owned = self._control.execute(text("""
            SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype='advisory'
                AND pid=pg_backend_pid() AND classid=0 AND objid=:key AND objsubid=1 AND granted)
        """), {'key':LOCK_KEY}).scalar_one()
        self._control.commit()
        if not owned:
            raise WorkerInterrupted('WORKER_LOCK_LOST')

    def recover_interrupted(self):
        self.verify_lock()
        with self.sessions() as db:
            jobs = db.scalars(self.scoped(select(AnalysisJob).where(
                AnalysisJob.status == 'running', AnalysisJob.worker_token.is_not(None)
            )).with_for_update(skip_locked=True)).all()
            recovered = 0
            for job in jobs:
                # Only worker-owned running jobs; never touch manual/legacy work.
                if db.get(VisionResult, job.id) is not None:
                    # Do not invent actual completion timestamps for inconsistent legacy data.
                    continue
                job.status = 'failed'
                job.finished_at = datetime.now(timezone.utc)
                job.error_message = 'WORKER_INTERRUPTED: previous worker stopped; submit a new job to retry'
                recovered += 1
            db.commit()
            return recovered

    def claim_next(self):
        self.verify_lock()
        with self.sessions() as db:
            query = self.scoped(select(AnalysisJob).where(AnalysisJob.status == 'queued'))
            if not self.allow_test:
                query = query.where(AnalysisJob.analysis_mode == 'production')
            query = query.order_by(AnalysisJob.created_at, AnalysisJob.id).with_for_update(skip_locked=True).limit(1)
            job = db.scalar(query)
            if job is None:
                db.rollback()
                return None
            if db.get(VisionResult, job.id) is not None:
                raise InputFailure('QUEUED_JOB_ALREADY_HAS_RESULT')
            video = db.get(Video, job.video_id)
            token, started = uuid4(), datetime.now(timezone.utc)
            job.status, job.worker_token, job.started_at = 'running', token, started
            job.finished_at, job.error_message = None, None
            claim = Claim(job.id, token, started, job.analysis_mode, video.storage_path, job.roi_snapshot, video.status, job.analysis_profile)
            db.commit()
            return claim

    def fail(self, claim, message):
        with self.sessions() as db:
            job = db.scalar(select(AnalysisJob).where(AnalysisJob.id==claim.job_id).with_for_update())
            if job and job.status == 'running' and job.worker_token == claim.token and db.get(VisionResult,job.id) is None:
                job.status = 'failed'
                job.finished_at = datetime.now(timezone.utc)
                job.error_message = message[:500]
                db.commit()

    @contextmanager
    def fail_stop_watchdog(self):
        # During compute only this thread uses the control connection. On DB
        # ownership loss terminate THIS dedicated worker, including inference.
        self.verify_lock()
        stopped = threading.Event()
        def monitor():
            while not stopped.wait(self.heartbeat_seconds):
                try:
                    self.verify_lock()
                except Exception:
                    self._lost.set()
                    os._exit(75)
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        try:
            yield
        finally:
            stopped.set()
            monitor_thread.join(timeout=max(10, self.heartbeat_seconds*3))
            if monitor_thread.is_alive():
                # Do not release the local slot while ownership is indeterminate.
                os._exit(75)

    def preflight(self, claim):
        if claim.analysis_profile not in ('fast', 'full'):
            raise InputFailure('INVALID_ANALYSIS_PROFILE')
        source = (self.base / claim.storage_path).resolve()
        if not source.is_relative_to((self.base/'uploads/videos').resolve()):
            raise InputFailure('VIDEO_PATH_NOT_ALLOWED')
        if claim.video_status != 'uploaded' or not source.is_file():
            raise InputFailure('VIDEO_FILE_UNAVAILABLE')
        try:
            state,_ = validate_roi(claim.roi)
        except InvalidResult:
            raise InputFailure('ROI_NOT_CONFIGURED_OR_INVALID') from None
        if state != 'approved' and not (claim.analysis_mode == 'test' and self.allow_test):
            raise InputFailure('ROI_NOT_APPROVED')
        if claim.analysis_mode not in ('production','test'):
            raise InputFailure('INVALID_ANALYSIS_MODE')
        info = self.inspect_video(source)
        if [info['width'],info['height']] != claim.roi['video_resolution']:
            raise InputFailure('ROI_RESOLUTION_MISMATCH')
        video_hash = sha256(source)
        if claim.roi.get('source_sha256',video_hash) != video_hash:
            raise InputFailure('ROI_VIDEO_HASH_MISMATCH')
        if not self.model_path.is_file() or self.model_path.stat().st_size == 0:
            raise InputFailure('LOCAL_MODEL_REQUIRED_NO_AUTO_DOWNLOAD')
        check_device(self.device)
        return source,info,video_hash

    def run_tracking(self, source: Path, out: Path):
        """Same adapter for durable jobs and bounded, DB-free smoke validation."""
        if self.track is not None:
            return self.track(source,out/'tracker','video'), {'backend':'mock','device':'mock',
                    'requested_device':self.device,'imgsz':self.imgsz,'batch':1,'half':False}
        os.environ['YOLO_AUTOINSTALL']='false'
        os.environ['YOLO_CONFIG_DIR']=str(out/'yolo-config')
        os.environ['MPLCONFIGDIR']=str(out/'mpl-config')
        from app.services import tracker
        summary = tracker.track_video(source,out/'tracker','video',model_path=self.model_path,
                                      imgsz=self.imgsz,device=self.device)
        return summary, summary['execution_settings']

    def run_once(self):
        claim = self.claim_next()
        if claim is None:
            return None
        stage = 'PREFLIGHT'
        try:
            source,info,video_hash = self.preflight(claim)
            root = self.base/'uploads/analysis'/str(claim.job_id)
            root.mkdir(parents=True,exist_ok=True)
            out = Path(tempfile.mkdtemp(prefix=f'{claim.token.hex}-',dir=root))
            roi_path=out/'roi_snapshot.json'
            roi_path.write_text(json.dumps(claim.roi,ensure_ascii=True,indent=2),encoding='utf-8')
            code_hashes={p.name:sha256(p) for p in (Path(__file__).parent/'services').glob('*.py') if p.name in (
                'tracker.py','pipeline.py','zone_analysis.py','optical_flow.py','individual_flow.py','bottleneck.py')}
            with self.fail_stop_watchdog(), (out/'worker.log').open('w',encoding='utf-8',buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
                stage = 'TRACKING'
                summary, ai_settings = self.run_tracking(source,out)
                device = ai_settings['device']
                detections=Path(summary['detections_csv']).resolve()
                if not detections.is_relative_to(out.resolve()) or not detections.is_file():
                    raise InputFailure('TRACKER_DETECTIONS_UNAVAILABLE')
                # Header-only detections are allowed; downstream CSV validation
                # decides whether that pipeline output represents a valid result.
                if summary['frame_count'] != info['frame_count']:
                    raise InputFailure('INCOMPLETE_VIDEO_DECODE')
                stage = 'ZONE_PIPELINE'
                from app.services.zone_analysis import load_zones_from_json
                if self.pipeline is None:
                    from app.services.pipeline import run_zone_pipeline
                    pipeline = run_zone_pipeline
                else:
                    pipeline = self.pipeline
                pipeline_args = (str(source),str(detections),load_zones_from_json(str(roi_path)),str(out/'pipeline'))
                # Preserve legacy four-argument FULL adapters.
                paths = (pipeline(*pipeline_args, analysis_profile='fast') if claim.analysis_profile == 'fast'
                         else pipeline(*pipeline_args))
                paths={kind:Path(path).resolve() for kind,path in paths.items()}
                if set(paths)!=set(SCHEMAS) or any(not p.is_relative_to(out.resolve()) for p in paths.values()):
                    raise InputFailure('PIPELINE_OUTPUT_PATH_NOT_ALLOWED')
                finished=datetime.now(timezone.utc)
                for name,digest in code_hashes.items():
                    if sha256(Path(__file__).parent/'services'/name)!=digest:
                        raise InputFailure('AI_SOURCE_CHANGED_DURING_EXECUTION')
                metadata=ExecutionMetadata(
                    analysis_started_at=claim.started_at,analysis_finished_at=finished,
                    model_version=self.model_path.name,algorithm_version=self.algorithm_version,
                    video_sha256=video_hash,fps=info['fps'],frame_count=info['frame_count'],
                    analysis_mode=claim.analysis_mode,analysis_profile=claim.analysis_profile,direction_quality='unvalidated',
                    settings={'analysis_mode':claim.analysis_mode,'analysis_profile':claim.analysis_profile,'device':device,'ai_source_sha256':code_hashes,
                              'model_sha256':sha256(self.model_path),**ai_settings},
                )
                (out/'execution_metadata.json').write_text(metadata.model_dump_json(indent=2),encoding='utf-8')
            self.verify_lock()
            stage = 'RESULT_STORAGE'
            with self.sessions() as db:
                storage.store_result(db,claim.job_id,paths,claim.roi,metadata,worker_token=claim.token)
            return {'job_id':str(claim.job_id),'status':'succeeded','analysis_profile':claim.analysis_profile,'output_dir':str(out)}
        except (Exception,KeyboardInterrupt) as exc:
            # Never put arbitrary exception strings/DB parameters in public state.
            if self.device != 'cpu' and is_cuda_oom(exc):
                code = 'CUDA_OUT_OF_MEMORY: reduce imgsz or model size; no automatic retry'
            else:
                code = str(exc) if isinstance(exc,(InputFailure,WorkerInterrupted,InferenceFailure)) else f'{stage}_FAILED ({type(exc).__name__})'
            self.fail(claim,code)
            if isinstance(exc,(KeyboardInterrupt,WorkerInterrupted)):
                raise
            return {'job_id':str(claim.job_id),'status':'failed','error':code}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',type=Path,required=True)
    parser.add_argument('--algorithm-version',required=True)
    parser.add_argument('--device',type=normalize_device,default='cpu',help='cpu (default), cuda:N or GPU index; no silent fallback')
    parser.add_argument('--imgsz',type=int,default=1920,help='Positive multiple of 32 (default: 1920)')
    parser.add_argument('--allow-test-analysis',action='store_true',help='Explicitly allow test jobs and candidate ROI')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--job-id',type=UUID,help='With --once, process/recover only this job')
    parser.add_argument('--poll-seconds',type=float,default=5)
    args=parser.parse_args()
    if args.poll_seconds<=0 or not math.isfinite(args.poll_seconds):
        parser.error('poll-seconds must be positive and finite')
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error('imgsz must be a positive multiple of 32')
    if not args.model_path.is_file() or args.model_path.stat().st_size == 0:
        parser.error('model-path must point to an existing local model; no automatic download')
    if args.job_id is not None and not args.once:
        parser.error('job-id requires --once')
    worker=Worker(args.model_path,args.algorithm_version,allow_test=args.allow_test_analysis,
                  scope_ids=[args.job_id] if args.job_id is not None else None,device=args.device,imgsz=args.imgsz)
    def interrupt(signum,frame):
        raise WorkerInterrupted('WORKER_STOP_REQUESTED')
    signal.signal(signal.SIGTERM,interrupt)
    try:
        with worker.exclusive():
            recovered=worker.recover_interrupted()
            print(f'Worker ready; interrupted jobs marked failed: {recovered}',flush=True)
            while True:
                result=worker.run_once()
                if result:
                    print(json.dumps(result),flush=True)
                    if result.get('error','').startswith('CUDA_OUT_OF_MEMORY'):
                        # Do not exhaust the remaining queue with the same GPU configuration.
                        return 1
                if args.once:
                    break
                if result is None:
                    time.sleep(args.poll_seconds)
    except (KeyboardInterrupt,WorkerInterrupted):
        print('Worker stopped; interrupted work is not automatically retried',flush=True)
    except WorkerBusy as exc:
        print(str(exc),file=sys.stderr)
        return 2
    except Exception as exc:
        print(f'Worker stopped due to {type(exc).__name__}; restart after checking database availability',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
