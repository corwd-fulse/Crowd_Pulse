import csv

import pandas as pd

from app.services.crowd_speed import compute_individual_speed, compute_zone_speed
from app.services.individual_flow import compute_individual_directions


def test_zone_speed_converts_pixels_to_meters_per_second():
    motion = pd.DataFrame(
        [
            {"sec": 0, "zone": "A", "angle_deg": 0.0, "magnitude_px": 10.0, "motion": 1.0, "low_confidence": False},
            {"sec": 0, "zone": "B", "angle_deg": None, "magnitude_px": None, "motion": None, "low_confidence": True},
        ]
    )
    out = compute_zone_speed(motion, fps=30.0, pixels_per_meter={"A": 100.0})

    row_a = out[out.zone == "A"].iloc[0]
    assert row_a.speed_mps == 3.0  # 10px/frame * 30frame/s / 100px/m = 3 m/s
    row_b = out[out.zone == "B"].iloc[0]
    assert pd.isna(row_b.speed_mps)
    assert row_b.low_confidence


def test_zone_speed_missing_calibration_is_low_confidence():
    motion = pd.DataFrame([{"sec": 0, "zone": "C", "angle_deg": 0.0, "magnitude_px": 5.0, "motion": 1.0, "low_confidence": False}])
    out = compute_zone_speed(motion, fps=30.0, pixels_per_meter={"A": 100.0})
    assert pd.isna(out.iloc[0].speed_mps)
    assert out.iloc[0].low_confidence


def test_individual_speed_converts_reliable_tracks_only(tmp_path):
    rows = [[f, f * 33, 1, 10 * f, 10, 10 * f + 10, 20, 0.9] for f in range(10)]
    csv_path = tmp_path / "det.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_ms", "track_id", "x1", "y1", "x2", "y2", "conf"])
        w.writerows(rows)

    individual = compute_individual_directions(str(csv_path), window_frames=5)
    out = compute_individual_speed(individual, fps=30.0, pixels_per_meter=10.0, window_frames=5)

    last = out[out.frame == out.frame.max()].iloc[0]
    assert last.reliable
    # 5프레임(=1/6초)에 50px 이동, pixels_per_meter=10 -> 5m / (1/6s) = 30 m/s
    assert abs(last.speed_mps - 30.0) < 0.5
