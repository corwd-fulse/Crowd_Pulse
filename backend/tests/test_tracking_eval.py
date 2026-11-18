import csv

from app.services.tracking_eval import evaluate_tracking


def _write_detections(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_ms", "track_id", "x1", "y1", "x2", "y2", "conf"])
        w.writerows(rows)


def test_stable_tracks_have_zero_suspected_switches(tmp_path):
    rows = [[f, f * 33, 1, 10 * f, 10, 10 * f + 10, 20, 0.9] for f in range(20)]
    csv_path = tmp_path / "det.csv"
    _write_detections(csv_path, rows)

    report = evaluate_tracking(str(csv_path))
    assert report["track_count"] == 1
    assert report["mean_track_length_frames"] == 20.0
    assert report["suspected_id_switch_ratio"] == 0.0


def test_all_untracked_detections_do_not_crash(tmp_path):
    # track_video가 ID를 못 붙인 경우(track_id 전부 빈 값) 흉내. 예전엔 lengths가 빈
    # Series가 되어 suspect_tracks/lengths.shape[0]에서 ZeroDivisionError가 났다.
    rows = [[f, f * 33, "", 10, 10, 20, 20, 0.9] for f in range(5)]
    csv_path = tmp_path / "det.csv"
    _write_detections(csv_path, rows)

    report = evaluate_tracking(str(csv_path))
    assert report["track_count"] == 0
    assert report["suspected_id_switch_ratio"] == 0.0
    assert report["mean_confidence"] == 0.9


def test_sudden_jump_counts_as_suspected_switch(tmp_path):
    rows = [[f, f * 33, 1, 10, 10, 20, 20, 0.9] for f in range(10)]
    rows += [[f, f * 33, 1, 500, 10, 510, 20, 0.9] for f in range(10, 20)]
    csv_path = tmp_path / "det.csv"
    _write_detections(csv_path, rows)

    report = evaluate_tracking(str(csv_path))
    assert report["suspected_id_switch_ratio"] == 1.0
