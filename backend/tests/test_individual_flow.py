import csv

import numpy as np
import pandas as pd

from app.services.individual_flow import compute_individual_directions, draw_direction_arrows

GRID_ANGLE_TOLERANCE = 1.0  # deg


def _write_detections(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_ms", "track_id", "x1", "y1", "x2", "y2", "conf"])
        w.writerows(rows)


def test_walking_right_gives_zero_degrees(tmp_path):
    rows = [[f, f * 33, 1, 10 * f, 10, 10 * f + 10, 20, 0.9] for f in range(10)]
    csv_path = tmp_path / "det.csv"
    _write_detections(csv_path, rows)

    out = compute_individual_directions(str(csv_path), window_frames=5)
    last = out[out.frame == out.frame.max()].iloc[0]
    assert last.reliable
    assert abs(last.angle_deg - 0.0) < GRID_ANGLE_TOLERANCE


def test_walking_down_gives_ninety_degrees(tmp_path):
    rows = [[f, f * 33, 2, 10, 10 * f, 20, 10 * f + 10, 0.9] for f in range(10)]
    csv_path = tmp_path / "det.csv"
    _write_detections(csv_path, rows)

    out = compute_individual_directions(str(csv_path), window_frames=5)
    last = out[out.frame == out.frame.max()].iloc[0]
    assert last.reliable
    assert abs(last.angle_deg - 90.0) < GRID_ANGLE_TOLERANCE


def test_id_switch_jump_marked_unreliable(tmp_path):
    rows = [[f, f * 33, 3, 10, 10, 20, 20, 0.9] for f in range(5)]
    rows += [[f, f * 33, 3, 210, 10, 220, 20, 0.9] for f in range(5, 10)]
    csv_path = tmp_path / "det.csv"
    _write_detections(csv_path, rows)

    out = compute_individual_directions(str(csv_path), window_frames=5)
    jump_row = out[out.frame == 5].iloc[0]
    assert not jump_row.reliable
    assert jump_row.angle_deg is None


def test_small_jitter_still_reliable_with_smoothing(tmp_path):
    # 전체적으로 오른쪽 이동이지만 프레임마다 +-2px 흔들림이 섞여 있다.
    jitter = [0, 2, -2, 1, -1, 2, -2, 0, 1, -1]
    rows = [[f, f * 33, 4, 10 * f + j, 10, 10 * f + j + 10, 20, 0.9] for f, j in enumerate(jitter)]
    csv_path = tmp_path / "det.csv"
    _write_detections(csv_path, rows)

    out = compute_individual_directions(str(csv_path), window_frames=5)
    last = out[out.frame == out.frame.max()].iloc[0]
    assert last.reliable
    assert abs(last.angle_deg - 0.0) < 15.0  # 창으로 묶어서 흔들림은 평균에 묻힘


def test_arrow_length_scales_with_speed():
    directions = pd.DataFrame(
        [
            {"track_id": 1, "angle_deg": 0.0, "speed_px": 1.0, "reliable": True},
            {"track_id": 2, "angle_deg": 0.0, "speed_px": 50.0, "reliable": True},
            {"track_id": 3, "angle_deg": 0.0, "speed_px": 0.0, "reliable": False},
        ]
    )
    positions = {1: (50, 50), 2: (150, 50), 3: (250, 50)}

    frame = np.zeros((100, 300, 3), dtype=np.uint8)
    out = draw_direction_arrows(frame, positions, directions)

    slow_tip_pixels = out[45:55, 70:90].sum()
    fast_tip_pixels = out[45:55, 200:230].sum()
    assert fast_tip_pixels > slow_tip_pixels  # 빠른 쪽 화살표가 더 멀리까지 그려져야 함
    assert out[42:58, 242:258].sum() > 0  # track 3(불가)는 원이 그려짐
