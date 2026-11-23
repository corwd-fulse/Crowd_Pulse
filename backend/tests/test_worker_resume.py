"""Worker._find_resumable_attempt()/_write_resume_manifest()의 DB 없이 돌 수 있는 단위 테스트.

이 두 메서드는 DB를 전혀 건드리지 않는다(self.preflight()도 순수 파일시스템/딥 로직) —
그래서 이 파일은 @pytest.mark.integration이 아니고, 실제 Postgres 없이도 그대로 돈다.
반대로 claim_next()의 started_at 보존, recover_interrupted()의 큐/실패 분기,
run_once()의 전체 재개 흐름은 DB(AnalysisJob row)가 반드시 필요해서 여기서 다루지 않는다 —
tests/test_worker.py에 @pytest.mark.integration으로 추가했고, 실제 Postgres가 있는
환경(docker compose)에서 돌려야 한다.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import cv2
import numpy as np
import pytest

from app.services.optical_flow import _Checkpoint, compute_checkpoint_fingerprint
from app.services.zone_analysis import Zone
from app.worker import Claim, MAX_RESUME_ATTEMPTS, RESUME_MANIFEST_NAME, Worker

ZONE = Zone("A", [(0, 0), (120, 0), (120, 120), (0, 120)])


def _write_video(path, n=20, fps=10, size=120):
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur(rng.integers(0, 255, (size, size), dtype=np.uint8), (15, 15), 0)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    for i in range(n):
        writer.write(np.roll(base, shift=i * 2, axis=1))
    writer.release()


def _sha256(path):
    import hashlib
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


class _Setup:
    """DB 없이 Worker.preflight()/_find_resumable_attempt()를 돌릴 수 있는 최소 환경."""

    def __init__(self, tmp_path, *, frame_stride=1, flow_method='full_frame', algorithm_version='algo-v1'):
        self.base = tmp_path
        (self.base / 'uploads/videos').mkdir(parents=True)
        self.video = self.base / 'uploads/videos/v.mp4'
        _write_video(self.video)
        self.video_hash = _sha256(self.video)
        model_path = self.base / 'model.pt'
        model_path.write_bytes(b'not-a-real-model')
        self.roi = {
            'video_resolution': [120, 120], 'approved_for_analysis': True, 'status': 'approved',
            'zones': [{'name': ZONE.name, 'polygon': [list(p) for p in ZONE.polygon]}],
            'source_sha256': self.video_hash,
        }
        self.worker = Worker(
            model_path, algorithm_version, base_dir=self.base, device='cpu',
            track=Mock(), pipeline=None, frame_stride=frame_stride, flow_method=flow_method,
            inspect_video=Mock(return_value={'width': 120, 'height': 120, 'fps': 10.0, 'frame_count': 20}),
        )
        self.job_id = uuid4()
        self.claim = Claim(self.job_id, uuid4(), datetime.now(timezone.utc), 'production',
                           'uploads/videos/v.mp4', self.roi, 'uploaded', 'full')
        self.root = self.base / 'uploads/analysis' / str(self.job_id)
        self.root.mkdir(parents=True)
        self.out = self.root / 'attempt-1'
        self.out.mkdir()
        self.detections = self.out / 'detections.csv'
        self.detections.write_text('frame,t_ms,track_id,x1,y1,x2,y2,conf\n', encoding='utf-8')
        self.checkpoint_path = self.root / 'flow_checkpoint.bin'
        self.code_hashes = {p.name: '0' * 64 for p in (model_path.parent).glob('*.py')}  # placeholder, overwritten below

    def code_hashes_now(self):
        from pathlib import Path as P
        import app.worker as worker_module
        from app.worker import sha256, TRACKED_SERVICE_FILES
        services_dir = P(worker_module.__file__).parent / 'services'
        return {p.name: sha256(p) for p in services_dir.glob('*.py') if p.name in TRACKED_SERVICE_FILES}

    def write_manifest(self, **overrides):
        flow_fp = compute_checkpoint_fingerprint(str(self.video), [ZONE], 10.0,
                                                  self.worker.frame_stride, self.worker.flow_method)
        manifest = dict(
            video_sha256=self.video_hash, roi_snapshot=self.roi,
            algorithm_version=self.worker.algorithm_version, analysis_mode='production',
            analysis_profile='full', frame_stride=self.worker.frame_stride,
            flow_method=self.worker.flow_method, imgsz=self.worker.imgsz, device=self.worker.device,
            code_hashes=self.code_hashes_now(), out_dir=str(self.out), detections_csv=str(self.detections),
            tracking_seconds=12.5, ai_settings={'device': 'cpu'}, checkpoint_path=str(self.checkpoint_path),
            flow_checkpoint_fingerprint=flow_fp, resume_count=0, created_at=datetime.now(timezone.utc).isoformat(),
        )
        manifest.update(overrides)
        (self.root / RESUME_MANIFEST_NAME).write_text(json.dumps(manifest), encoding='utf-8')
        return manifest

    def write_checkpoint(self, fingerprint=None):
        fp = fingerprint or compute_checkpoint_fingerprint(str(self.video), [ZONE], 10.0,
                                                            self.worker.frame_stride, self.worker.flow_method)
        _Checkpoint(self.checkpoint_path).save(fp, 7, {(0, 'A'): [1.0, 2.0, 1, False]},
                                               {'A': (1.0, 2.0, False)}, np.zeros((4, 4), dtype=np.uint8))


def test_no_manifest_means_not_resumable(tmp_path):
    s = _Setup(tmp_path)
    plan, reason = s.worker._find_resumable_attempt(s.claim)
    assert plan is None and reason is None


def test_valid_manifest_and_checkpoint_is_resumable(tmp_path):
    s = _Setup(tmp_path)
    s.write_manifest()
    s.write_checkpoint()
    plan, reason = s.worker._find_resumable_attempt(s.claim)
    assert reason is None
    assert plan is not None
    assert plan.out_dir == s.out
    assert plan.detections_csv == s.detections
    assert plan.tracking_seconds == 12.5
    assert plan.ai_settings == {'device': 'cpu'}


@pytest.mark.parametrize("field,bad_value", [
    ("video_sha256", "0" * 64),
    ("algorithm_version", "some-other-version"),
    ("analysis_mode", "test"),
    ("analysis_profile", "fast"),
    ("frame_stride", 99),
    ("flow_method", "roi_crop_128"),
    ("imgsz", 640),
    ("device", "cuda:0"),
])
def test_fingerprint_mismatch_refuses_resume_with_clear_reason(tmp_path, field, bad_value):
    s = _Setup(tmp_path)
    s.write_manifest(**{field: bad_value})
    s.write_checkpoint()
    plan, reason = s.worker._find_resumable_attempt(s.claim)
    assert plan is None
    assert "RESUME_FINGERPRINT_MISMATCH" in reason and field in reason


def test_code_hash_mismatch_refuses_resume(tmp_path):
    s = _Setup(tmp_path)
    tampered = s.code_hashes_now()
    tampered["optical_flow.py"] = "deadbeef" * 8
    s.write_manifest(code_hashes=tampered)
    s.write_checkpoint()
    plan, reason = s.worker._find_resumable_attempt(s.claim)
    assert plan is None
    assert "code_hashes" in reason


def test_roi_mismatch_refuses_resume(tmp_path):
    s = _Setup(tmp_path)
    different_roi = dict(s.roi)
    different_roi["zones"] = [{"name": "B", "polygon": [[0, 0], [10, 0], [10, 10]]}]
    s.write_manifest(roi_snapshot=different_roi)
    s.write_checkpoint()
    plan, reason = s.worker._find_resumable_attempt(s.claim)
    assert plan is None and "roi_snapshot" in reason


def test_missing_out_dir_refuses_resume(tmp_path):
    s = _Setup(tmp_path)
    s.write_manifest()
    s.write_checkpoint()
    import shutil
    shutil.rmtree(s.out)
    plan, reason = s.worker._find_resumable_attempt(s.claim)
    assert plan is None and reason == "RESUME_ATTEMPT_FILES_MISSING"


def test_missing_detections_csv_refuses_resume(tmp_path):
    s = _Setup(tmp_path)
    s.write_manifest()
    s.write_checkpoint()
    s.detections.unlink()
    plan, reason = s.worker._find_resumable_attempt(s.claim)
    assert plan is None and reason == "RESUME_ATTEMPT_FILES_MISSING"


def test_resume_attempt_limit_is_enforced(tmp_path):
    s = _Setup(tmp_path)
    s.write_manifest(resume_count=MAX_RESUME_ATTEMPTS)
    s.write_checkpoint()
    plan, reason = s.worker._find_resumable_attempt(s.claim)
    assert plan is None
    assert "RESUME_ATTEMPT_LIMIT_EXCEEDED" in reason


def test_corrupted_checkpoint_refuses_resume_with_reason(tmp_path):
    s = _Setup(tmp_path)
    s.write_manifest()
    s.write_checkpoint()
    s.checkpoint_path.write_bytes(b"garbage, not a real checkpoint")
    plan, reason = s.worker._find_resumable_attempt(s.claim)
    assert plan is None
    assert "RESUME_CHECKPOINT_INVALID" in reason


def test_unreadable_manifest_json_refuses_resume_with_reason(tmp_path):
    s = _Setup(tmp_path)
    (s.root / RESUME_MANIFEST_NAME).write_text("{not valid json", encoding="utf-8")
    plan, reason = s.worker._find_resumable_attempt(s.claim)
    assert plan is None
    assert "RESUME_MANIFEST_UNREADABLE" in reason


def test_write_resume_manifest_round_trips_through_find_resumable_attempt(tmp_path):
    """_write_resume_manifest()가 실제로 쓰는 필드들이 _find_resumable_attempt()가
    기대하는 필드와 정확히 맞물리는지 — 두 메서드가 따로 수정되다 어긋나지 않게 확인."""
    s = _Setup(tmp_path)
    flow_fp = compute_checkpoint_fingerprint(str(s.video), [ZONE], 10.0,
                                             s.worker.frame_stride, s.worker.flow_method)
    s.worker._write_resume_manifest(
        s.claim, s.root, s.out, s.detections, s.video_hash, s.code_hashes_now(),
        12.5, {'device': 'cpu'}, s.checkpoint_path, flow_fp,
    )
    s.write_checkpoint(fingerprint=flow_fp)
    plan, reason = s.worker._find_resumable_attempt(s.claim)
    assert reason is None and plan is not None
    assert plan.tracking_seconds == 12.5
