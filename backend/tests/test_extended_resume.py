"""확장 분석 이어하기: DM-Count 윈도우 체크포인트 + 단계별 진행 기록."""
import csv
import json
import shutil

import cv2
import numpy as np
import pandas as pd
import pytest

from app.services import density_estimation
from app.services.density_estimation import compute_dmcount_estimates
from app.services.pipeline import DMCOUNT_CHECKPOINT_NAME, EXTENDED_PROGRESS_NAME, run_extended_zone_pipeline
from app.services.zone_analysis import Zone

ZONE = Zone("A", [(0, 0), (120, 0), (120, 120), (0, 120)])


class Crash(Exception):
    pass


def _write_video(path, n_frames=60, fps=10, size=120):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    for i in range(n_frames):
        writer.write(np.full((size, size, 3), (i * 4) % 255, dtype=np.uint8))
    writer.release()


def _write_detections(path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_ms", "track_id", "x1", "y1", "x2", "y2", "conf"])
        w.writerows([[f, f * 100, 1, 10 + f, 10, 20 + f, 20, 0.9] for f in range(60)])


class CountingDensity:
    """프레임 밝기에 따라 값이 달라지는 가짜 추정기. crash_after번째 호출에서 예외를 던져
    worker가 죽는 상황을 흉내 낸다."""

    def __init__(self, crash_after=None):
        self.calls = 0
        self.crash_after = crash_after

    def __call__(self, frame):
        if self.crash_after is not None and self.calls >= self.crash_after:
            raise Crash()
        self.calls += 1
        return np.full((12, 12), float(frame.mean()) / 100, dtype=np.float32), 1.0


def _run(video, density_fn, **kwargs):
    kwargs.setdefault("window_s", 1)
    return compute_dmcount_estimates(str(video), [ZONE], fps=10, density_fn=density_fn, **kwargs)


def test_dmcount_resumes_after_crash_and_matches_uninterrupted_run(tmp_path):
    video = tmp_path / "v.mp4"
    _write_video(video)
    expected = _run(video, CountingDensity(), heatmap_dir=tmp_path / "ref_heat")

    ckpt = tmp_path / "work" / "ckpt.json"
    heat = tmp_path / "work" / "heat"
    with pytest.raises(Crash):
        _run(video, CountingDensity(crash_after=7), heatmap_dir=heat, checkpoint_path=ckpt)
    state = json.loads(ckpt.read_text())
    assert state["next_window_start"] == 2  # 3장씩, 7번째 호출 직전 = 윈도우 2개 완료

    resumed_fn = CountingDensity()
    result = _run(video, resumed_fn, heatmap_dir=heat, checkpoint_path=ckpt)

    assert resumed_fn.calls == 3 * 4  # 남은 윈도우(2~5초)만 추론
    pd.testing.assert_frame_equal(result.drop(columns="density_map_path"),
                                  expected.drop(columns="density_map_path"))
    assert result["density_map_path"].map(lambda p: p.startswith(str(heat))).all()
    assert not ckpt.exists()  # 정상 종료 시 삭제


def test_dmcount_checkpoint_with_different_settings_restarts(tmp_path):
    video = tmp_path / "v.mp4"
    _write_video(video)
    ckpt = tmp_path / "ckpt.json"
    with pytest.raises(Crash):
        _run(video, CountingDensity(crash_after=7), checkpoint_path=ckpt)

    fn = CountingDensity()
    _run(video, fn, checkpoint_path=ckpt, gamma=1.5)
    assert fn.calls == 3 * 6  # 서명 불일치 → 처음부터


def test_dmcount_checkpoint_survives_moving_to_another_machine(tmp_path):
    """경로가 바뀌어도(다른 컴퓨터로 복사) 이어진다 — 서명에 경로/mtime이 없다."""
    server = tmp_path / "server"
    server.mkdir()
    _write_video(server / "v.mp4")
    with pytest.raises(Crash):
        _run(server / "v.mp4", CountingDensity(crash_after=10), heatmap_dir=server / "heat",
             checkpoint_path=server / "ckpt.json")

    local = tmp_path / "local"
    shutil.copytree(server, local)
    fn = CountingDensity()
    result = _run(local / "v.mp4", fn, heatmap_dir=local / "heat", checkpoint_path=local / "ckpt.json")
    assert fn.calls == 3 * 3
    assert sorted(result["window_start"]) == [0, 1, 2, 3, 4, 5]
    assert result["density_map_path"].map(lambda p: p.startswith(str(local / "heat"))).all()


def test_dmcount_checkpoint_missing_heatmap_restarts(tmp_path):
    video = tmp_path / "v.mp4"
    _write_video(video)
    heat, ckpt = tmp_path / "heat", tmp_path / "ckpt.json"
    with pytest.raises(Crash):
        _run(video, CountingDensity(crash_after=7), heatmap_dir=heat, checkpoint_path=ckpt)
    (heat / "window_0.png").unlink()
    fn = CountingDensity()
    _run(video, fn, heatmap_dir=heat, checkpoint_path=ckpt)
    assert fn.calls == 3 * 6


_ORIGINAL_DMCOUNT = density_estimation.compute_dmcount_estimates


def _patch_dmcount(monkeypatch, density_fn):
    original = _ORIGINAL_DMCOUNT

    def fake(*args, **kwargs):
        kwargs["density_fn"] = density_fn
        kwargs["window_s"] = 1
        return original(*args, **kwargs)

    monkeypatch.setattr(density_estimation, "compute_dmcount_estimates", fake)


def test_extended_pipeline_resumes_dmcount_only_on_another_machine(tmp_path, monkeypatch):
    """base 지표가 끝난 뒤 DM-Count 중에 끊기면, 다른 컴퓨터에서 DM-Count만 이어간다. 처음 계산이
    YOLO 잔여를 구했으므로 추적 CSV(det.csv)는 같이 옮긴다 — 없으면 아래 테스트처럼 멈춘다."""
    server = tmp_path / "server"
    server.mkdir()
    _write_video(server / "v.mp4")
    _write_detections(server / "det.csv")
    weights = server / "w.pth"
    weights.write_bytes(b"not-really-weights")

    _patch_dmcount(monkeypatch, CountingDensity(crash_after=7))
    with pytest.raises(Crash):
        run_extended_zone_pipeline(str(server / "v.mp4"), str(server / "det.csv"), [ZONE], str(server / "out"),
                                   dmcount_model_path=str(weights), resumable=True)
    progress = json.loads((server / "out" / EXTENDED_PROGRESS_NAME).read_text())
    assert set(progress["stages"]) == {"base"}
    assert (server / "out" / DMCOUNT_CHECKPOINT_NAME).is_file()

    # 다른 컴퓨터: out/ + 영상 + 가중치 + 추적 CSV 복사
    local = tmp_path / "local"
    local.mkdir()
    shutil.copytree(server / "out", local / "out")
    shutil.copy(server / "v.mp4", local / "v.mp4")
    shutil.copy(weights, local / "w.pth")
    shutil.copy(server / "det.csv", local / "det.csv")
    fn = CountingDensity()
    _patch_dmcount(monkeypatch, fn)
    paths = run_extended_zone_pipeline(str(local / "v.mp4"), str(local / "det.csv"), [ZONE], str(local / "out"),
                                       dmcount_model_path=str(local / "w.pth"), resumable=True)

    assert fn.calls == 3 * 4
    assert set(paths) == {"dwell_times", "counter_flow", "queue_scores", "density_heatmap",
                          "tracking_quality", "dmcount_estimates"}
    assert all(str(p).startswith(str(local / "out")) for p in paths.values())
    estimates = pd.read_csv(paths["dmcount_estimates"])
    assert len(estimates) == 6
    assert estimates["yolo_residual_count"].notna().all()
    assert not (local / "out" / DMCOUNT_CHECKPOINT_NAME).exists()

    # 다 끝난 뒤 다시 부르면 아무것도 재계산하지 않는다.
    again = CountingDensity()
    _patch_dmcount(monkeypatch, again)
    assert run_extended_zone_pipeline(str(local / "v.mp4"), str(local / "det.csv"), [ZONE], str(local / "out"),
                                      dmcount_model_path=str(local / "w.pth"), resumable=True) == paths
    assert again.calls == 0


def test_extended_pipeline_recomputes_base_when_output_changed(tmp_path):
    _write_video(tmp_path / "v.mp4")
    _write_detections(tmp_path / "det.csv")
    out = tmp_path / "out"
    run_extended_zone_pipeline(str(tmp_path / "v.mp4"), str(tmp_path / "det.csv"), [ZONE], str(out), resumable=True)
    original = (out / "dwell_times.csv").read_text()
    (out / "dwell_times.csv").write_text("corrupted")
    run_extended_zone_pipeline(str(tmp_path / "v.mp4"), str(tmp_path / "det.csv"), [ZONE], str(out), resumable=True)
    assert (out / "dwell_times.csv").read_text() == original


def test_extended_pipeline_default_writes_no_progress_files(tmp_path):
    _write_video(tmp_path / "v.mp4")
    _write_detections(tmp_path / "det.csv")
    run_extended_zone_pipeline(str(tmp_path / "v.mp4"), str(tmp_path / "det.csv"), [ZONE], str(tmp_path / "out"))
    assert not (tmp_path / "out" / EXTENDED_PROGRESS_NAME).exists()


def test_fp32_checkpoint_can_continue_in_fp16_and_records_precisions(tmp_path):
    video = tmp_path / "v.mp4"
    _write_video(video)
    ckpt = tmp_path / "ckpt.json"
    with pytest.raises(Crash):
        _run(video, CountingDensity(crash_after=7), checkpoint_path=ckpt)

    fn = CountingDensity()
    result = _run(video, fn, checkpoint_path=ckpt, half=True)
    assert fn.calls == 3 * 4  # precision is not part of the signature: it continues
    assert result.attrs["windows_by_precision"] == {"fp32": 2, "fp16": 4}


def test_fp16_without_cuda_is_rejected(tmp_path):
    video = tmp_path / "v.mp4"
    _write_video(video)
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"x")
    with pytest.raises(ValueError, match="CUDA"):
        compute_dmcount_estimates(str(video), [ZONE], fps=10, model_path=weights, device="cpu", half=True)


def test_extended_pipeline_records_dmcount_precision(tmp_path, monkeypatch):
    _write_video(tmp_path / "v.mp4")
    _write_detections(tmp_path / "det.csv")
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"x")
    _patch_dmcount(monkeypatch, CountingDensity())
    run_extended_zone_pipeline(str(tmp_path / "v.mp4"), str(tmp_path / "det.csv"), [ZONE], str(tmp_path / "out"),
                               dmcount_model_path=str(weights), resumable=True, dmcount_half=True)
    progress = json.loads((tmp_path / "out" / EXTENDED_PROGRESS_NAME).read_text())
    assert progress["meta"]["dmcount:dmcount_qnrf:residual"] == {"windows_by_precision": {"fp16": 6}}


def test_residual_checkpoint_without_detections_stops_instead_of_restarting(tmp_path):
    """YOLO 잔여까지 구하던 계산을 추적 CSV 없이 이어가려 하면 조용히 처음부터 돌리지 않고 멈춘다."""
    video = tmp_path / "v.mp4"
    _write_video(video)
    _write_detections(tmp_path / "det.csv")
    ckpt = tmp_path / "ckpt.json"
    with pytest.raises(Crash):
        _run(video, CountingDensity(crash_after=7), checkpoint_path=ckpt, detections_csv=tmp_path / "det.csv")
    fn = CountingDensity()
    with pytest.raises(ValueError, match="video_detections"):
        _run(video, fn, checkpoint_path=ckpt)
    assert fn.calls == 0
    resumed = CountingDensity()
    _run(video, resumed, checkpoint_path=ckpt, detections_csv=tmp_path / "det.csv")
    assert resumed.calls == 3 * 4


def test_uncovered_mask_excludes_padded_boxes():
    boxes = np.array([[10, 10, 20, 20]], dtype=np.float32)  # 프레임 120x120 → map 12x12 (1/10)
    mask = density_estimation._uncovered_mask(boxes, (120, 120), (12, 12))
    # 10% 여유(1px)를 넣으면 9..21px → 칸 0..3 (끝 칸 포함)
    assert (~mask).sum() == 16
    assert not mask[0:4, 0:4].any() and mask[4:, :].all()
    assert density_estimation._uncovered_mask(np.empty((0, 4), np.float32), (120, 120), (12, 12)).all()


def test_residual_is_density_outside_yolo_boxes(tmp_path):
    video = tmp_path / "v.mp4"
    _write_video(video)
    _write_detections(tmp_path / "det.csv")
    plain = _run(video, CountingDensity())
    fused = _run(video, CountingDensity(), detections_csv=tmp_path / "det.csv")
    assert "yolo_residual_count" not in plain.columns
    pd.testing.assert_frame_equal(fused.drop(columns="yolo_residual_count"), plain)
    # 박스가 모든 프레임에 하나씩 있으므로 잔여는 전체보다 작고 0보다 크다.
    assert ((fused["yolo_residual_count"] < fused["estimated_count"]) | (fused["estimated_count"] == 0)).all()
    assert (fused["yolo_residual_count"] >= 0).all()


def test_dmcount_csv_with_and_without_residual_validates(tmp_path):
    from app.services.vision_extended_csv import InvalidExtendedResult, _read_csv_rows

    base = "window_start,zone,estimated_count,inference_ms,model_version,density_map_path"
    old = tmp_path / "old.csv"
    old.write_text(base + "\n0,A,2.5,10.0,dmcount_qnrf,\n", encoding="utf-8")
    new = tmp_path / "new.csv"
    new.write_text(base + ",yolo_residual_count\n0,A,2.5,10.0,dmcount_qnrf,,0.75\n", encoding="utf-8")
    assert "yolo_residual_count" not in _read_csv_rows(old, "dmcount_estimates")[0]
    assert _read_csv_rows(new, "dmcount_estimates")[0]["yolo_residual_count"] == 0.75
    bad = tmp_path / "bad.csv"
    bad.write_text(base + ",yolo_residual_count\n0,A,2.5,10.0,dmcount_qnrf,,-1\n", encoding="utf-8")
    with pytest.raises(InvalidExtendedResult):
        _read_csv_rows(bad, "dmcount_estimates")
