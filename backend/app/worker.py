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

import pandas as pd
from sqlalchemy import select, text, update
from sqlalchemy.orm import sessionmaker

from app.database import engine
from app.models import AnalysisJob, Video, VisionResult
from app.services import vision_results as storage
from app.services import vision_extended_results as extended_storage
from app.services.vision_csv import ExecutionMetadata, InvalidResult, SCHEMAS, validate_roi
from app.services.inference_runtime import normalize_device, check_device, InferenceFailure, is_cuda_oom

LOCK_KEY = 19201112

# Optical Flow가 중간 체크포인트를 남길 수 있는 서비스 모듈들 — AI_SOURCE_CHANGED_DURING_EXECUTION
# 검사와 resume_manifest의 code_hashes가 항상 이 동일한 목록을 공유한다(목록이 서로 다르면
# "코드가 바뀌었는지" 판단이 재개 경로와 일반 경로에서 어긋날 수 있다).
TRACKED_SERVICE_FILES = (
    'tracker.py', 'pipeline.py', 'zone_analysis.py', 'optical_flow.py', 'individual_flow.py',
    'bottleneck.py', 'hybrid_flow.py', 'bottleneck_diagnostics.py', 'zone_replay.py',
    'roi_geometry.py', 'transition_artifact.py',
)

RESUME_MANIFEST_NAME = 'resume_manifest.json'
# 체크포인트에서 재개를 시도했는데 또 끊기는 상황이 끝없이 반복되는 걸 막는 상한선. 재개
# 시도 자체가 아니라 "재개가 필요했던 횟수"를 센다 — 매 restart마다 하나씩 늘어난다.
MAX_RESUME_ATTEMPTS = 5


@dataclass(frozen=True)
class ResumePlan:
    """_find_resumable_attempt()가 "재개해도 안전하다"고 판단했을 때 돌려주는 정보.
    run_once()가 이걸 받으면 tracking을 다시 돌리지 않고 이 out_dir/detections_csv를 그대로
    쓰고, checkpoint_path로 zone_pipeline을 호출해 Optical Flow만 이어서 계산한다."""
    out_dir: Path
    detections_csv: Path
    checkpoint_path: Path
    tracking_seconds: float
    ai_settings: dict
    manifest_path: Path
    resume_count: int


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
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


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
                 extended_pipeline=None, pixels_per_meter=None, enable_extended=False, frame_stride=1,
                 dmcount_model_path=None,
                 inspect_video=video_info, heartbeat_seconds=2.0, scope_ids=None, device='cpu', imgsz=1920,
                 flow_method='full_frame'):
        self.bind = bind
        self.sessions = sessions or sessionmaker(bind, expire_on_commit=False)
        self.base = Path(base_dir or storage.BASE_DIR).resolve()
        self.model_path = Path(model_path).resolve()
        self.device = normalize_device(device)
        if flow_method not in ('full_frame', 'roi_crop_128'):
            raise ValueError('Invalid flow method')
        if flow_method == 'roi_crop_128' and (self.device == 'cpu' or not allow_test or not scope_ids):
            raise ValueError('ROI crop requires GPU, test permission and explicit job scope')
        self.flow_method = flow_method
        if type(imgsz) is not int or imgsz <= 0 or imgsz % 32:
            raise ValueError('imgsz must be a positive multiple of 32')
        self.imgsz = imgsz
        if not algorithm_version.strip():
            raise ValueError('Explicit algorithm version is required')
        self.algorithm_version = algorithm_version
        self.allow_test = allow_test
        self.track = track
        self.pipeline = pipeline
        self.extended_pipeline = extended_pipeline
        self.pixels_per_meter = pixels_per_meter
        # Opt-in: existing callers/tests that don't know about the extended pipeline must see
        # byte-for-byte the same behavior as before it existed. Only --enable-extended-analysis
        # (or passing enable_extended=True directly) turns this on.
        self.enable_extended = enable_extended
        # None(기본) = 이 job에서 DM-Count 안 돌림. 느린 CPU 추론에 정확도 미검증이라
        # (density_estimation.py 모듈 docstring 참고) 명시적으로 체크포인트 경로를 줄 때만 켜진다.
        self.dmcount_model_path = dmcount_model_path
        if type(frame_stride) is not int or frame_stride < 1:
            raise ValueError('frame_stride must be a positive int')
        self.frame_stride = frame_stride
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

    def _find_resumable_attempt(self, claim) -> tuple[ResumePlan | None, str | None]:
        """이 claim(job)에 대해 안전하게 재개할 수 있는 이전 시도가 남아있는지 확인한다.

        반환: (ResumePlan, None) — 재개 가능. (None, None) — 재개할 게 원래 없음(일반
        job이거나 아직 tracking도 못 끝낸 상태) → 기존처럼 처리. (None, reason) — 재개를
        시도할 흔적은 있었지만 유효하지 않음 → 호출 측이 reason을 명확한 실패 사유로 남겨야
        한다(임의로 조용히 새로 돌리면 안 됨 — recover_interrupted()에서만 이 구분이 중요하다;
        run_once()에서는 reason이 있어도 그냥 새로 도는 것으로 안전하게 처리한다).

        video 해시/ROI/모델 등은 self.preflight(claim)을 그대로 재사용해 검증한다 — 별도
        로직을 또 만들면 두 경로(최초 실행/재개)의 검증 기준이 갈라질 위험이 있다.
        """
        root = self.base / 'uploads/analysis' / str(claim.job_id)
        manifest_path = root / RESUME_MANIFEST_NAME
        if not manifest_path.is_file():
            return None, None
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            return None, f'RESUME_MANIFEST_UNREADABLE ({type(exc).__name__})'
        try:
            source, info, video_hash = self.preflight(claim)
        except InputFailure as exc:
            return None, f'RESUME_PREFLIGHT_FAILED ({exc})'
        code_hashes = {p.name: sha256(p) for p in (Path(__file__).parent / 'services').glob('*.py')
                      if p.name in TRACKED_SERVICE_FILES}
        current = dict(
            video_sha256=video_hash, roi_snapshot=claim.roi, algorithm_version=self.algorithm_version,
            analysis_mode=claim.analysis_mode, analysis_profile=claim.analysis_profile,
            frame_stride=self.frame_stride, flow_method=self.flow_method, imgsz=self.imgsz,
            device=self.device, code_hashes=code_hashes,
        )
        for field, value in current.items():
            if manifest.get(field) != value:
                return None, f'RESUME_FINGERPRINT_MISMATCH ({field})'
        out_dir, detections_csv = Path(manifest['out_dir']), Path(manifest['detections_csv'])
        if not out_dir.is_dir() or not detections_csv.is_file():
            return None, 'RESUME_ATTEMPT_FILES_MISSING'
        resume_count = int(manifest.get('resume_count', 0))
        if resume_count >= MAX_RESUME_ATTEMPTS:
            return None, f'RESUME_ATTEMPT_LIMIT_EXCEEDED ({resume_count} >= {MAX_RESUME_ATTEMPTS})'
        checkpoint_path = Path(manifest['checkpoint_path'])
        from app.services.optical_flow import inspect_checkpoint
        status = inspect_checkpoint(checkpoint_path, manifest['flow_checkpoint_fingerprint'])
        if not status.valid:
            return None, f'RESUME_CHECKPOINT_INVALID ({status.reason})'
        return ResumePlan(out_dir, detections_csv, checkpoint_path,
                          float(manifest.get('tracking_seconds', 0.0)), dict(manifest.get('ai_settings') or {}),
                          manifest_path, resume_count), None

    def _write_resume_manifest(self, claim, root: Path, out: Path, detections: Path,
                               video_hash: str, code_hashes: dict, tracking_seconds: float, ai_settings: dict,
                               checkpoint_path: Path, flow_checkpoint_fingerprint: str) -> None:
        manifest = dict(
            video_sha256=video_hash, roi_snapshot=claim.roi, algorithm_version=self.algorithm_version,
            analysis_mode=claim.analysis_mode, analysis_profile=claim.analysis_profile,
            frame_stride=self.frame_stride, flow_method=self.flow_method, imgsz=self.imgsz,
            device=self.device, code_hashes=code_hashes,
            out_dir=str(out), detections_csv=str(detections), tracking_seconds=tracking_seconds,
            ai_settings=ai_settings,
            checkpoint_path=str(checkpoint_path), flow_checkpoint_fingerprint=flow_checkpoint_fingerprint,
            resume_count=0, created_at=datetime.now(timezone.utc).isoformat(),
        )
        tmp = (root / RESUME_MANIFEST_NAME).with_suffix('.json.tmp')
        tmp.write_text(json.dumps(manifest), encoding='utf-8')
        os.replace(tmp, root / RESUME_MANIFEST_NAME)

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
                video = db.get(Video, job.video_id)
                pseudo_claim = Claim(job.id, job.worker_token, job.started_at, job.analysis_mode,
                                     video.storage_path, job.roi_snapshot, video.status, job.analysis_profile)
                plan, reason = self._find_resumable_attempt(pseudo_claim)
                if plan is not None:
                    # 재개 대상으로 전환: 같은 job_id, 같은 started_at(연속성) — 소유권만
                    # 비워서 claim_next()가 새 worker_token으로 다시 집도록 한다. 재시도
                    # 횟수를 여기서 먼저 올려 적어둔다 — 이번 재개도 또 끊기면 다음
                    # recover_interrupted()가 그 값을 보고 상한을 넘었는지 판단한다.
                    manifest = json.loads(plan.manifest_path.read_text(encoding='utf-8'))
                    manifest['resume_count'] = plan.resume_count + 1
                    plan.manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
                    job.status, job.worker_token = 'queued', None
                    print(f'job {job.id}: resuming from checkpoint (attempt {plan.resume_count + 1}'
                          f'/{MAX_RESUME_ATTEMPTS})', flush=True)
                else:
                    job.status = 'failed'
                    job.finished_at = datetime.now(timezone.utc)
                    job.error_message = (
                        'WORKER_INTERRUPTED: previous worker stopped; submit a new job to retry'
                        if reason is None else f'WORKER_INTERRUPTED_NO_RESUME: {reason}'
                    )[:500]
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
            job.status, job.worker_token = 'running', token
            # started_at은 최초 시작 시각 그대로 둔다(연속성) — 재개된 job이 아니라면
            # None이었을 것이므로 이번이 처음이고, 그때만 새로 채운다.
            if job.started_at is None:
                job.started_at = started
            job.finished_at, job.error_message = None, None
            claim = Claim(job.id, token, job.started_at, job.analysis_mode, video.storage_path, job.roi_snapshot, video.status, job.analysis_profile)
            db.commit()
            return claim

    def make_flow_progress_reporter(self, claim, root: Path):
        """Optical Flow 단계 진행률/ETA를 파일(root/flow_progress.json)과 DB에 남긴다.

        진행률 보고 자체가 실패해도(예: 그 순간 DB 연결 문제) job을 절대 실패시키지
        않는다 — 보고는 부가 기능이고, 실제 분석 결과 저장과는 독립이다.
        """
        progress_path = root / 'flow_progress.json'
        started = time.perf_counter()

        def report(frame_idx: int, frame_count: int):
            now = datetime.now(timezone.utc)
            elapsed = time.perf_counter() - started
            percent = round(100 * frame_idx / frame_count, 1) if frame_count else None
            eta_seconds = None
            if frame_count and frame_idx and elapsed > 0:
                rate = frame_idx / elapsed
                eta_seconds = max(frame_count - frame_idx, 0) / rate if rate else None
            payload = dict(frame_idx=frame_idx, frame_count=frame_count, percent=percent,
                          eta_seconds=eta_seconds, updated_at=now.isoformat())
            try:
                progress_path.write_text(json.dumps(payload), encoding='utf-8')
            except OSError:
                pass
            try:
                with self.sessions() as db:
                    db.execute(update(AnalysisJob).where(AnalysisJob.id == claim.job_id).values(
                        progress_percent=percent, progress_eta_seconds=eta_seconds,
                        progress_updated_at=now))
                    db.commit()
            except Exception:
                pass
        return report

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
        if self.flow_method == 'roi_crop_128' and claim.analysis_mode != 'test':
            raise InputFailure('EXPERIMENT_REQUIRES_TEST_JOB')
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
        root = None
        try:
            source,info,video_hash = self.preflight(claim)
            root = self.base/'uploads/analysis'/str(claim.job_id)
            root.mkdir(parents=True,exist_ok=True)
            # resuming=True only when a previous attempt for this exact job already
            # completed tracking and left a checkpoint whose fingerprint (video/ROI/code/
            # settings) matches this run. See _find_resumable_attempt()'s docstring for what
            # "matches" means and ResumePlan for what gets reused.
            plan, _ = self._find_resumable_attempt(claim) if self.pipeline is None else (None, None)
            resuming = plan is not None
            out = plan.out_dir if resuming else Path(tempfile.mkdtemp(prefix=f'{claim.token.hex}-',dir=root))
            roi_path=out/'roi_snapshot.json'
            if not resuming:
                roi_path.write_text(json.dumps(claim.roi,ensure_ascii=True,indent=2),encoding='utf-8')
            code_hashes={p.name:sha256(p) for p in (Path(__file__).parent/'services').glob('*.py') if p.name in TRACKED_SERVICE_FILES}
            with self.fail_stop_watchdog(), (out/'worker.log').open('a' if resuming else 'w',encoding='utf-8',buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
                stage = 'TRACKING'
                if resuming:
                    print(f'resuming job {claim.job_id} from Optical Flow checkpoint '
                          f'{plan.checkpoint_path} (resume attempt {plan.resume_count}/{MAX_RESUME_ATTEMPTS}); '
                          f'tracking already completed in a previous attempt, reusing {out}', flush=True)
                    tracking_seconds = plan.tracking_seconds
                    ai_settings = plan.ai_settings
                    device = ai_settings.get('device')
                    detections = plan.detections_csv
                else:
                    tracking_started = time.perf_counter()
                    summary, ai_settings = self.run_tracking(source,out)
                    tracking_seconds = time.perf_counter() - tracking_started
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
                    # frame_stride is only safe to pass to the real pipeline function. A
                    # caller-supplied self.pipeline may be a legacy four-positional-argument
                    # adapter (see tests/test_worker.py's `worker` fixture) that predates this
                    # option and has no **kwargs to absorb it.
                    full_kwargs = {
                        'frame_stride': self.frame_stride,
                        # root (stable per-job dir), not out (fresh per-attempt tempdir): a
                        # checkpoint must survive being read back within this same attempt,
                        # and the per-attempt dir is only created once per run_once() call
                        # anyway, so this is just "the one checkpoint file for this attempt".
                        'checkpoint_path': str(root / 'flow_checkpoint.bin'),
                        'progress_callback': self.make_flow_progress_reporter(claim, root),
                    }
                else:
                    pipeline = self.pipeline
                    full_kwargs = {}
                zones = load_zones_from_json(str(roi_path))
                if not resuming and self.pipeline is None and claim.analysis_profile == 'full':
                    # Write the resume manifest *before* the (potentially multi-hour) Optical
                    # Flow call, not after — it must exist while that call is running, since
                    # that is exactly the stage a crash/restart would interrupt.
                    from app.services.optical_flow import compute_checkpoint_fingerprint
                    flow_fp = compute_checkpoint_fingerprint(str(source), zones, info['fps'],
                                                             self.frame_stride, self.flow_method)
                    self._write_resume_manifest(claim, root, out, detections, video_hash, code_hashes,
                                                tracking_seconds, ai_settings, root/'flow_checkpoint.bin', flow_fp)
                pipeline_args = (str(source),str(detections),zones,str(out/'pipeline'))
                # Preserve legacy four-argument FULL adapters.
                if self.flow_method == 'roi_crop_128':
                    paths = pipeline(*pipeline_args, analysis_profile=claim.analysis_profile,
                                     flow_method=self.flow_method, **full_kwargs)
                else:
                    paths = (pipeline(*pipeline_args, analysis_profile='fast') if claim.analysis_profile == 'fast'
                             else pipeline(*pipeline_args, **full_kwargs))
                details_path = out/'pipeline'/'analysis_details.json'
                details = json.loads(details_path.read_text(encoding='utf-8')) if details_path.is_file() else {}
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
                              'model_sha256':sha256(self.model_path),**ai_settings,
                              'requested_flow_method':self.flow_method,
                              'tracking_seconds':tracking_seconds,'resumed_from_checkpoint':resuming, **details},
                )
                (out/'execution_metadata.json').write_text(metadata.model_dump_json(indent=2),encoding='utf-8')
            self.verify_lock()
            stage = 'RESULT_STORAGE'
            with self.sessions() as db:
                storage.store_result(db,claim.job_id,paths,claim.roi,metadata,worker_token=claim.token)
            # Job reached a terminal, successfully-stored state in this same call — any
            # resume manifest for it is no longer meaningful (the checkpoint itself was
            # already self-deleted by compute_zone_motion on its own successful return).
            (root/RESUME_MANIFEST_NAME).unlink(missing_ok=True)
            # Extended metrics (dwell/counter-flow/queue/heatmap/tracking-quality) are a
            # best-effort bonus on top of the core FR-ZON result above, which has already
            # committed. A failure here must never flip a succeeded job to failed, so it is
            # logged to worker.log and swallowed instead of re-raised. Opt-in only
            # (self.enable_extended) so every existing caller/test keeps its exact prior
            # behavior unless it asks for this.
            if claim.analysis_profile == 'full' and self.enable_extended:
                try:
                    if self.extended_pipeline is None:
                        from app.services.pipeline import run_extended_zone_pipeline
                        extended = run_extended_zone_pipeline
                    else:
                        extended = self.extended_pipeline
                    # Reuse what run_zone_pipeline() already computed instead of re-running
                    # Optical Flow (the most expensive step) a second time over the whole video.
                    extended_paths = extended(
                        str(source),str(detections),zones,str(out/'extended'),
                        pixels_per_meter=self.pixels_per_meter,
                        zone_timeseries=pd.read_csv(paths['crowd_observations']),
                        motion_timeseries=pd.read_csv(paths['flow']),
                        individual_directions=pd.read_csv(paths['individual_directions']),
                        dmcount_model_path=self.dmcount_model_path,
                        dmcount_device=self.device,
                    )
                    extended_paths = {kind:Path(path).resolve() for kind,path in extended_paths.items()}
                    if any(not p.is_relative_to(out.resolve()) for p in extended_paths.values()):
                        raise InputFailure('EXTENDED_PIPELINE_OUTPUT_PATH_NOT_ALLOWED')
                    with self.sessions() as db:
                        extended_storage.store_extended_result(db,claim.job_id,extended_paths,
                                                                pixels_per_meter=self.pixels_per_meter)
                except Exception as exc:
                    with (out/'worker.log').open('a',encoding='utf-8') as log:
                        log.write(f'EXTENDED_PIPELINE_FAILED ({type(exc).__name__}): {exc}\n')
            return {'job_id':str(claim.job_id),'status':'succeeded','analysis_profile':claim.analysis_profile,'output_dir':str(out)}
        except (Exception,KeyboardInterrupt) as exc:
            # Never put arbitrary exception strings/DB parameters in public state.
            if self.device != 'cpu' and is_cuda_oom(exc):
                code = 'CUDA_OUT_OF_MEMORY: reduce imgsz or model size; no automatic retry'
            else:
                code = str(exc) if isinstance(exc,(InputFailure,WorkerInterrupted,InferenceFailure)) else f'{stage}_FAILED ({type(exc).__name__})'
            self.fail(claim,code)
            # This run_once() call is what decided the job's fate (failed, terminal) — via a
            # caught exception, not an abrupt process death — so any resume manifest is now
            # stale regardless of outcome. (A true abrupt death never reaches this line at
            # all, which is exactly the case recover_interrupted() exists to find later.)
            if root is not None:
                (root/RESUME_MANIFEST_NAME).unlink(missing_ok=True)
            if isinstance(exc,(KeyboardInterrupt,WorkerInterrupted)):
                raise
            return {'job_id':str(claim.job_id),'status':'failed','error':code}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',type=Path,required=True)
    parser.add_argument('--algorithm-version',required=True)
    parser.add_argument('--device',type=normalize_device,default='cpu',help='cpu (default), cuda:N or GPU index; no silent fallback')
    parser.add_argument('--imgsz',type=int,default=1920,help='Positive multiple of 32 (default: 1920)')
    parser.add_argument('--flow-method',choices=('full_frame','roi_crop_128'),default='full_frame',
                        help='Experimental crop requires GPU, --allow-test-analysis and --once --job-id; FAST skips flow')
    parser.add_argument('--allow-test-analysis',action='store_true',help='Explicitly allow test jobs and candidate ROI')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--job-id',type=UUID,help='With --once, process/recover only this job')
    parser.add_argument('--poll-seconds',type=float,default=5)
    parser.add_argument('--pixels-per-meter',type=float,default=None,
                        help='Camera calibration for extended crowd-speed metrics (m/s); omit to skip speed conversion')
    parser.add_argument('--enable-extended-analysis',action='store_true',
                        help='Also compute and store dwell/counter-flow/queue/heatmap/tracking-quality metrics (FULL profile only)')
    parser.add_argument('--frame-stride',type=int,default=1,
                        help='Run Optical Flow every N frames instead of every frame on long FULL videos (default 1: unchanged, every frame)')
    parser.add_argument('--dmcount-model-path',type=Path,default=None,
                        help='DM-Count pretrained checkpoint; omit (default) to skip this slow, accuracy-unverified reference metric entirely')
    args=parser.parse_args()
    if args.poll_seconds<=0 or not math.isfinite(args.poll_seconds):
        parser.error('poll-seconds must be positive and finite')
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error('imgsz must be a positive multiple of 32')
    if args.pixels_per_meter is not None and args.pixels_per_meter <= 0:
        parser.error('pixels-per-meter must be positive')
    if args.pixels_per_meter is not None and not args.enable_extended_analysis:
        parser.error('pixels-per-meter has no effect without --enable-extended-analysis')
    if args.frame_stride < 1:
        parser.error('frame-stride must be a positive int')
    if not args.model_path.is_file() or args.model_path.stat().st_size == 0:
        parser.error('model-path must point to an existing local model; no automatic download')
    if args.dmcount_model_path is not None and not args.enable_extended_analysis:
        parser.error('dmcount-model-path has no effect without --enable-extended-analysis')
    if args.dmcount_model_path is not None and (not args.dmcount_model_path.is_file() or args.dmcount_model_path.stat().st_size == 0):
        parser.error('dmcount-model-path must point to an existing local checkpoint; no automatic download')
    if args.job_id is not None and not args.once:
        parser.error('job-id requires --once')
    if args.flow_method == 'roi_crop_128' and (args.device == 'cpu' or not args.allow_test_analysis or args.job_id is None):
        parser.error('roi_crop_128 requires GPU, --allow-test-analysis and --once --job-id')
    worker=Worker(args.model_path,args.algorithm_version,allow_test=args.allow_test_analysis,
                  scope_ids=[args.job_id] if args.job_id is not None else None,device=args.device,imgsz=args.imgsz,
                  pixels_per_meter=args.pixels_per_meter,enable_extended=args.enable_extended_analysis,
                  frame_stride=args.frame_stride,flow_method=args.flow_method,
                  dmcount_model_path=args.dmcount_model_path)
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
