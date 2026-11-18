import csv
import json

import cv2
import numpy as np
import pandas as pd

from app.services import pipeline as pipeline_module
from app.services.pipeline import run_extended_zone_pipeline
from app.services.vision_csv import SCHEMAS
from app.services.zone_analysis import Zone


def _write_video(path, size=120, n=10, fps=10):
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.integers(0, 255, (size, size), dtype=np.uint8), (15, 15), 0)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    for i in range(n):
        writer.write(np.roll(base, shift=i * 2, axis=1))
    writer.release()


def _write_detections(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_ms", "track_id", "x1", "y1", "x2", "y2", "conf"])
        w.writerows(rows)


def test_extended_pipeline_produces_new_files_without_touching_core_schemas(tmp_path):
    video_path = tmp_path / "v.mp4"
    _write_video(video_path)
    zone = Zone("A", [(0, 0), (120, 0), (120, 120), (0, 120)])
    rows = [[f, f * 100, 1, 10 + f, 10, 20 + f, 20, 0.9] for f in range(10)]
    det_path = tmp_path / "det.csv"
    _write_detections(det_path, rows)

    paths = run_extended_zone_pipeline(str(video_path), str(det_path), [zone], str(tmp_path / "out"))

    assert set(paths) == {"dwell_times", "counter_flow", "queue_scores", "density_heatmap", "tracking_quality"}
    assert not set(paths) & set(SCHEMAS)  # 기존 5개 결과 계약과 겹치지 않아야 함

    quality = json.loads(open(paths["tracking_quality"]).read())
    assert quality["track_count"] == 1


def test_extended_pipeline_adds_speed_files_only_with_calibration(tmp_path):
    video_path = tmp_path / "v.mp4"
    _write_video(video_path)
    zone = Zone("A", [(0, 0), (120, 0), (120, 120), (0, 120)])
    rows = [[f, f * 100, 1, 10 + f, 10, 20 + f, 20, 0.9] for f in range(10)]
    det_path = tmp_path / "det.csv"
    _write_detections(det_path, rows)

    paths = run_extended_zone_pipeline(
        str(video_path), str(det_path), [zone], str(tmp_path / "out"), pixels_per_meter=50.0
    )

    assert "crowd_speed_zone" in paths
    assert "crowd_speed_individual" in paths


def test_extended_pipeline_reuses_precomputed_frames_without_recomputing(tmp_path, monkeypatch):
    # worker.py가 핵심 파이프라인 결과(zone_timeseries/motion_timeseries/individual_directions)를
    # 넘겨주면, 이 함수는 (특히 Optical Flow처럼) 그걸 다시 계산하면 안 된다 — 장시간 영상에서
    # 제일 비싼 연산을 두 번 돌리게 되는 회귀를 막는 테스트.
    video_path = tmp_path / "v.mp4"
    _write_video(video_path)
    zone = Zone("A", [(0, 0), (120, 0), (120, 120), (0, 120)])
    rows = [[f, f * 100, 1, 10 + f, 10, 20 + f, 20, 0.9] for f in range(10)]
    det_path = tmp_path / "det.csv"
    _write_detections(det_path, rows)

    def boom(*a, **k):
        raise AssertionError("recomputed a value that was already supplied")

    monkeypatch.setattr(pipeline_module, "compute_zone_motion", boom)
    monkeypatch.setattr(pipeline_module, "compute_individual_directions", boom)
    monkeypatch.setattr(pipeline_module, "compute_zone_timeseries", boom)

    zone_ts = pd.DataFrame([{"sec": 0, "zone": "A", "count": 1.0, "inflow": 0, "outflow": 0}])
    motion_ts = pd.DataFrame([{"sec": 0, "zone": "A", "angle_deg": 0.0, "magnitude_px": 1.0, "motion": 1.0, "low_confidence": False}])
    individual = pd.DataFrame(columns=["frame", "t_ms", "track_id", "angle_deg", "speed_px", "reliable"])

    paths = run_extended_zone_pipeline(
        str(video_path), str(det_path), [zone], str(tmp_path / "out"),
        zone_timeseries=zone_ts, motion_timeseries=motion_ts, individual_directions=individual,
    )
    assert set(paths) == {"dwell_times", "counter_flow", "queue_scores", "density_heatmap", "tracking_quality"}
