"""Optical Flow 단계의 checkpoint/resume + 진행률 콜백 결정성 테스트.

범위: app/services/optical_flow.py의 checkpoint_path/progress_callback 자체, 그리고
app/services/pipeline.py가 compute_zone_motion() 중간 예외를 같은 호출 안에서 한 번
재시도하는 동작. worker.py의 job 재시도 정책(동일 job 재제출 금지)과는 무관하다 —
여기서 다루는 건 같은 프로세스·같은 파이프라인 호출 안에서의 재개뿐이다.
"""
import csv
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services import optical_flow
from app.services.optical_flow import CheckpointStatus, _Checkpoint
from app.services.pipeline import _run_zone_pipeline
from app.services.zone_analysis import Zone

ZONE = Zone("A", [(0, 0), (120, 0), (120, 120), (0, 120)])


def _write_video(path, n=40, fps=10, size=120):
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur(rng.integers(0, 255, (size, size), dtype=np.uint8), (15, 15), 0)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    for i in range(n):
        writer.write(np.roll(base, shift=i * 2, axis=1))
    writer.release()


def _write_detections(path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_ms", "track_id", "x1", "y1", "x2", "y2", "conf"])


def test_resumed_run_matches_single_uninterrupted_run(tmp_path):
    video = tmp_path / "v.mp4"
    _write_video(video)
    full = optical_flow.compute_zone_motion(str(video), [ZONE], fps=10, frame_stride=2)

    ckpt = tmp_path / "ckpt.bin"
    orig_read = cv2.VideoCapture.read
    calls = {"n": 0}

    class _Boom(RuntimeError):
        pass

    def flaky(self):
        calls["n"] += 1
        if calls["n"] == 15:
            raise _Boom("simulated mid-run failure")
        return orig_read(self)

    cv2.VideoCapture.read = flaky
    try:
        with pytest.raises(_Boom):
            optical_flow.compute_zone_motion(str(video), [ZONE], fps=10, frame_stride=2,
                                             checkpoint_path=str(ckpt), checkpoint_every_frames=5)
    finally:
        cv2.VideoCapture.read = orig_read
    assert ckpt.exists(), "checkpoint must survive an interrupted run"

    resumed = optical_flow.compute_zone_motion(str(video), [ZONE], fps=10, frame_stride=2,
                                               checkpoint_path=str(ckpt), checkpoint_every_frames=5)
    assert not ckpt.exists(), "checkpoint must be cleared after success"
    full = full.reset_index(drop=True)
    resumed = resumed.reset_index(drop=True)
    assert full["sec"].tolist() == resumed["sec"].tolist()
    assert full["zone"].tolist() == resumed["zone"].tolist()
    assert full["low_confidence"].tolist() == resumed["low_confidence"].tolist()
    for col in ("angle_deg", "magnitude_px", "motion"):
        a, b = full[col].to_numpy(dtype=float), resumed[col].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.nanmax(np.abs(a - b)[~both_nan]) < 1e-9 if (~both_nan).any() else True


def test_progress_callback_reports_monotonic_progress(tmp_path):
    video = tmp_path / "v.mp4"
    _write_video(video, n=60)
    calls = []
    optical_flow.compute_zone_motion(str(video), [ZONE], fps=10, frame_stride=1,
                                     checkpoint_every_frames=10,
                                     progress_callback=lambda f, n: calls.append((f, n)))
    assert len(calls) >= 2
    frames = [c[0] for c in calls]
    assert frames == sorted(frames)
    assert all(total == 0 or total >= frame for frame, total in calls)


def test_pipeline_resumes_once_from_checkpoint_after_flow_exception(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    _write_video(video)
    det = tmp_path / "det.csv"
    _write_detections(det)
    ckpt = tmp_path / "flow_checkpoint.bin"

    real_compute = optical_flow.compute_zone_motion
    state = {"calls": 0}

    def flaky_compute(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("simulated optical-flow stage crash")
        return real_compute(*args, **kwargs)

    monkeypatch.setattr("app.services.pipeline.compute_zone_motion", flaky_compute)

    paths = _run_zone_pipeline(str(video), str(det), [ZONE], str(tmp_path / "out"),
                               checkpoint_path=str(ckpt))
    assert state["calls"] == 2, "pipeline should retry exactly once using the checkpoint path"
    assert Path(paths["flow"]).exists()


@pytest.mark.parametrize("corruption", ["truncated", "bad_magic", "flipped_byte", "empty"])
def test_corrupted_checkpoint_is_rejected_with_a_clear_reason(tmp_path, corruption):
    ckpt_path = tmp_path / "ckpt.bin"
    good = _Checkpoint(ckpt_path)
    good.save("fp", 7, {(0, "A"): [1.0, 2.0, 1, False]}, {"A": (1.0, 2.0, False)},
              np.zeros((4, 4), dtype=np.uint8))
    raw = ckpt_path.read_bytes()
    if corruption == "truncated":
        ckpt_path.write_bytes(raw[: len(raw) // 2])
    elif corruption == "bad_magic":
        ckpt_path.write_bytes(b"NOTAMAGIC" + raw[9:])
    elif corruption == "flipped_byte":
        mid = len(raw) // 2
        ckpt_path.write_bytes(raw[:mid] + bytes([raw[mid] ^ 0xFF]) + raw[mid + 1:])
    elif corruption == "empty":
        ckpt_path.write_bytes(b"")

    state, status = _Checkpoint(ckpt_path).load("fp")
    assert state is None
    assert status.valid is False
    assert status.reason  # 구체적인 이유가 있어야 함(조용히 무시만 하면 안 됨)

    # compute_zone_motion 자체도 손상된 checkpoint를 보고 그냥 멈추거나 틀린 값을 내지 않고,
    # 처음부터 안전하게 다시 돈다.
    video = tmp_path / "v.mp4"
    _write_video(video)
    out = optical_flow.compute_zone_motion(str(video), [ZONE], fps=10,
                                           checkpoint_path=str(ckpt_path))
    assert not out.empty
    assert not ckpt_path.exists(), "성공적으로 끝났으면 checkpoint는 정리되어야 한다"


def test_inspect_checkpoint_matches_load_semantics(tmp_path):
    ckpt_path = tmp_path / "ckpt.bin"
    assert optical_flow.inspect_checkpoint(ckpt_path, "fp") == CheckpointStatus(False, None, None)

    _Checkpoint(ckpt_path).save("fp-a", 42, {}, {}, np.zeros((2, 2), dtype=np.uint8))
    mismatched = optical_flow.inspect_checkpoint(ckpt_path, "fp-b")
    assert mismatched.valid is False and "fingerprint" in mismatched.reason

    matched = optical_flow.inspect_checkpoint(ckpt_path, "fp-a")
    assert matched == CheckpointStatus(True, None, 42)


def test_checkpoint_save_survives_being_killed_mid_write(tmp_path):
    """저장 중간에 프로세스가 강제종료돼도(디스크에 보이는 건 이전 checkpoint 그대로),
    이후 load()가 깨진 걸 보지 않고 이전 정상 checkpoint를 그대로 읽는지 확인한다.
    실제 kill -9(복구 코드가 전혀 못 끼어드는 상황)를 서브프로세스로 흉내낸다."""
    ckpt_path = tmp_path / "ckpt.bin"
    good = _Checkpoint(ckpt_path)
    good.save("fp", 10, {(0, "A"): [1.0, 2.0, 1, False]}, {"A": (1.0, 2.0, False)},
              np.full((4, 4), 5, dtype=np.uint8))
    good_bytes = ckpt_path.read_bytes()

    backend_root = str(Path(__file__).resolve().parents[1])
    script = tmp_path / "slow_save.py"
    script.write_text(textwrap.dedent(f"""
        import builtins, sys, time
        sys.path.insert(0, {backend_root!r})
        import numpy as np
        from app.services.optical_flow import _Checkpoint

        real_open = builtins.open
        def slow_open(path, mode='r', *a, **kw):
            f = real_open(path, mode, *a, **kw)
            if 'tmp' in str(path) and 'w' in mode:
                print('OPENED_TMP', flush=True)
                time.sleep(10)
            return f
        builtins.open = slow_open

        ckpt = _Checkpoint({str(ckpt_path)!r})
        ckpt.save('fp', 999, {{(0, 'A'): [9.0, 9.0, 9, False]}}, {{'A': (9.0, 9.0, False)}},
                  np.full((4, 4), 9, dtype=np.uint8))
        print('SHOULD_NOT_REACH_HERE', flush=True)
    """), encoding="utf-8")

    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 15
    saw_opened_tmp = False
    line = proc.stdout.readline()
    while line:
        if "OPENED_TMP" in line:
            saw_opened_tmp = True
            break
        if time.time() > deadline:
            break
        line = proc.stdout.readline()
    assert saw_opened_tmp, "subprocess never reached the mid-write point to kill at"
    proc.kill()  # SIGKILL on Linux / TerminateProcess on Windows — no cleanup code runs
    proc.wait(timeout=10)

    assert ckpt_path.read_bytes() == good_bytes, "the prior valid checkpoint must be untouched"
    state, status = _Checkpoint(ckpt_path).load("fp")
    assert status.valid is True and state[0] == 10, "load() must still see the old, valid checkpoint"


def test_pipeline_does_not_swallow_a_second_failure(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    _write_video(video)
    det = tmp_path / "det.csv"
    _write_detections(det)

    def always_fails(*args, **kwargs):
        raise RuntimeError("persistent failure")

    monkeypatch.setattr("app.services.pipeline.compute_zone_motion", always_fails)

    with pytest.raises(RuntimeError, match="persistent failure"):
        _run_zone_pipeline(str(video), str(det), [ZONE], str(tmp_path / "out"),
                           checkpoint_path=str(tmp_path / "ckpt.bin"))
