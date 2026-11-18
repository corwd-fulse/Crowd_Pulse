import csv

from app.services.density_heatmap import compute_density_heatmap


def _write_detections(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_ms", "track_id", "x1", "y1", "x2", "y2", "conf"])
        w.writerows(rows)


def test_detections_land_in_expected_cell(tmp_path):
    # 1000x1000 프레임, grid_size=10 -> 셀 크기 100px. 바닥 중심이 (950, 950) 근처면 셀 (9,9).
    rows = [[f, f * 100, 1, 940, 900, 960, 950, 0.9] for f in range(10)]
    csv_path = tmp_path / "det.csv"
    _write_detections(csv_path, rows)

    out = compute_density_heatmap(str(csv_path), frame_width=1000, frame_height=1000, fps=30.0, grid_size=10, window_s=10)
    row = out[(out.cell_x == 9) & (out.cell_y == 9)].iloc[0]
    assert row.window_start == 0
    assert row["count"] > 0
    assert len(out) == 1


def test_two_zones_get_separate_cells(tmp_path):
    rows = [[f, f * 100, 1, 10, 10, 20, 20, 0.9] for f in range(5)]  # 좌상단
    rows += [[f, f * 100, 2, 940, 940, 960, 960, 0.9] for f in range(5)]  # 우하단
    csv_path = tmp_path / "det.csv"
    _write_detections(csv_path, rows)

    out = compute_density_heatmap(str(csv_path), frame_width=1000, frame_height=1000, fps=30.0, grid_size=10, window_s=10)
    assert len(out) == 2
