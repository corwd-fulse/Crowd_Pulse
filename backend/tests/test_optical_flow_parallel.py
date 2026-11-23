"""멀티코어(workers>1) Optical Flow가 단일 코어와 같은 결과를 내는지 검증한다.

PARALLEL_SEGMENT_FRAMES를 작게 줄여 짧은 합성 영상도 여러 구간으로 나뉘게 한다. 구간 경계는
부모 프로세스에서 계산해 자식에게 넘기므로, 자식(spawn)이 모듈을 새로 import해도 패치한
값이 그대로 적용된다.
"""
import cv2
import numpy as np
import pandas as pd
import pytest

from app.services import optical_flow
from app.services.optical_flow import compute_checkpoint_fingerprint, compute_zone_motion, inspect_checkpoint
from app.services.zone_analysis import Zone

ZONES = [Zone("A", [(0, 0), (60, 0), (60, 120), (0, 120)]),
         Zone("B", [(60, 0), (120, 0), (120, 120), (60, 120)])]


def _write_video(path, n=95, fps=10, size=120):
    rng = np.random.default_rng(2)
    base = cv2.GaussianBlur(rng.integers(0, 255, (size, size), dtype=np.uint8), (15, 15), 0)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    for i in range(n):
        # 구간마다 움직임 방향/속도가 달라야 경계 처리 오류가 결과 차이로 드러난다.
        writer.write(np.roll(base, shift=(i * 2 if i < 50 else 100 - i), axis=1 if i % 30 < 20 else 0))
    writer.release()


def _assert_same(serial: pd.DataFrame, parallel: pd.DataFrame):
    assert list(serial.columns) == list(parallel.columns)
    assert len(serial) == len(parallel)
    pd.testing.assert_frame_equal(serial.reset_index(drop=True), parallel.reset_index(drop=True),
                                  check_exact=False, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("frame_stride", [1, 3])
def test_parallel_matches_single_core(tmp_path, monkeypatch, frame_stride):
    monkeypatch.setattr(optical_flow, "PARALLEL_SEGMENT_FRAMES", 20)
    video = tmp_path / "v.mp4"
    _write_video(video)

    serial = compute_zone_motion(str(video), ZONES, fps=10, frame_stride=frame_stride)
    parallel = compute_zone_motion(str(video), ZONES, fps=10, frame_stride=frame_stride, workers=3)

    assert serial["low_confidence"].eq(False).any()  # 실제로 움직임이 계산된 초가 있어야 의미가 있다
    _assert_same(serial, parallel)


def test_parallel_resumes_finished_segments_and_reports_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(optical_flow, "PARALLEL_SEGMENT_FRAMES", 20)
    video = tmp_path / "v.mp4"
    _write_video(video)
    ckpt = tmp_path / "flow.ckpt"
    expected = compute_zone_motion(str(video), ZONES, fps=10)

    real_segment = optical_flow._flow_segment
    calls = []

    def crash_after_two(pool_results):
        for i, item in enumerate(pool_results):
            if i == 2:
                raise RuntimeError("simulated crash")
            yield item

    class CrashingPool:
        def __init__(self, processes):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def imap_unordered(self, fn, tasks):
            return crash_after_two(map(fn, tasks))

    class Ctx:
        Pool = CrashingPool

    monkeypatch.setattr(optical_flow.multiprocessing, "get_context", lambda method: Ctx)
    with pytest.raises(RuntimeError, match="simulated crash"):
        compute_zone_motion(str(video), ZONES, fps=10, checkpoint_path=str(ckpt), workers=3)

    fp = compute_checkpoint_fingerprint(str(video), ZONES, 10, 1, "full_frame", parallel=True)
    status = inspect_checkpoint(ckpt, fp)
    assert status.valid and status.frame_idx > 0  # worker.py의 재개 판단에도 쓰이는 형식

    def counting_segment(task):
        calls.append(task[0])
        return real_segment(task)

    class InlinePool(CrashingPool):
        def imap_unordered(self, fn, tasks):
            return map(counting_segment, tasks)

    class Ctx2:
        Pool = InlinePool

    monkeypatch.setattr(optical_flow.multiprocessing, "get_context", lambda method: Ctx2)
    progress = []
    resumed = compute_zone_motion(str(video), ZONES, fps=10, checkpoint_path=str(ckpt), workers=3,
                                  progress_callback=lambda done, total: progress.append((done, total)))

    assert 0 not in calls and 1 not in calls  # 끝난 두 구간은 다시 계산하지 않는다
    assert progress == sorted(progress) and progress[-1][0] == 94  # 프레임 0은 비교 기준이라 제외
    assert not ckpt.exists()
    _assert_same(expected, resumed)


def test_seek_mismatch_falls_back_to_single_core(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(optical_flow, "PARALLEL_SEGMENT_FRAMES", 20)
    video = tmp_path / "v.mp4"
    _write_video(video)
    expected = compute_zone_motion(str(video), ZONES, fps=10)

    real_segment = optical_flow._flow_segment

    def broken_seek(task):
        seg_id, acc, processed, anchor, next_anchor = real_segment(task)
        return seg_id, acc, processed, ("wrong" if seg_id == 2 else anchor), next_anchor

    class InlinePool:
        def __init__(self, processes):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def imap_unordered(self, fn, tasks):
            return map(broken_seek, tasks)

    class Ctx:
        Pool = InlinePool

    monkeypatch.setattr(optical_flow.multiprocessing, "get_context", lambda method: Ctx)
    out = compute_zone_motion(str(video), ZONES, fps=10, workers=3)

    assert "recomputing on a single core" in capsys.readouterr().out
    _assert_same(expected, out)


def test_short_video_or_single_worker_uses_single_core(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    _write_video(video, n=30)

    def no_pool(method):
        raise AssertionError("parallel path must not run")

    monkeypatch.setattr(optical_flow.multiprocessing, "get_context", no_pool)
    # 기본 구간 길이(3000)보다 짧은 영상은 나눌 의미가 없어 단일 코어로 계산한다.
    compute_zone_motion(str(video), ZONES, fps=10, workers=4)
    with pytest.raises(ValueError):
        compute_zone_motion(str(video), ZONES, fps=10, workers=0)
