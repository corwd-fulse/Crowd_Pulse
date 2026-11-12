import csv

from app.services.zone_analysis import Zone, compute_zone_timeseries, load_zones_from_json

ZONE_A = Zone("A", [(0, 0), (100, 0), (100, 100), (0, 100)])
ZONE_B = Zone("B", [(200, 0), (300, 0), (300, 100), (200, 100)])


def _write_detections(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_ms", "track_id", "x1", "y1", "x2", "y2", "conf"])
        w.writerows(rows)


def test_outflow_from_zone_a(tmp_path):
    # track 1: 구역 A(x~15)에서 밖(x~155)으로 10프레임에 걸쳐 걸어나간다(실제 추적처럼 점진적 이동).
    rows = [[f, f * 33, 1, 10 + 14 * f, 10, 20 + 14 * f, 20, 0.9] for f in range(10)]
    csv_path = tmp_path / "det.csv"
    _write_detections(csv_path, rows)

    out = compute_zone_timeseries(str(csv_path), [ZONE_A, ZONE_B], fps=30.0)
    assert out[(out.zone == "A")]["outflow"].sum() == 1
    assert out[(out.zone == "B")]["inflow"].sum() == 0


def test_boundary_jitter_is_filtered(tmp_path):
    # track 2: A -> 밖 -> A 를 1프레임씩만 왔다갔다 (min_dwell=3 미달) -> 이벤트 없어야 함
    states = ["A", "A", None, "A", None, "A", "A"]
    rows = []
    for f, state in enumerate(states):
        if state == "A":
            rows.append([f, f * 33, 2, 10, 10, 20, 20, 0.9])
        else:
            rows.append([f, f * 33, 2, 150, 10, 160, 20, 0.9])
    csv_path = tmp_path / "det.csv"
    _write_detections(csv_path, rows)

    out = compute_zone_timeseries(str(csv_path), [ZONE_A, ZONE_B], fps=30.0)
    assert out[(out.zone == "A")]["inflow"].sum() == 0
    assert out[(out.zone == "A")]["outflow"].sum() == 0


def test_inflow_to_zone_b(tmp_path):
    # track 3: 구역 밖(x~155)에서 구역 B(x~215)로 10프레임에 걸쳐 점진적으로 걸어들어간다.
    rows = [[f, f * 33, 3, 150 + 10 * f, 10, 160 + 10 * f, 20, 0.9] for f in range(10)]
    csv_path = tmp_path / "det.csv"
    _write_detections(csv_path, rows)

    out = compute_zone_timeseries(str(csv_path), [ZONE_A, ZONE_B], fps=30.0)
    assert out[(out.zone == "B")]["inflow"].sum() == 1


def test_stationary_track_near_boundary_is_not_counted(tmp_path):
    # track 4: 경계선(x=100) 바로 위에서 거의 안 움직이며 박스 노이즈로 A/밖을 오간다.
    # (E05_001 실데이터에서 track 706이 23초간 12번 "횡단"으로 잘못 잡혔던 패턴)
    xs = [92, 93, 108, 109, 93, 94, 108, 109, 93, 108, 109, 93]
    rows = [[f, f * 33, 4, x, 10, x + 10, 20, 0.9] for f, x in enumerate(xs)]
    csv_path = tmp_path / "det.csv"
    _write_detections(csv_path, rows)

    out = compute_zone_timeseries(str(csv_path), [ZONE_A, ZONE_B], fps=30.0)
    assert out[(out.zone == "A")]["inflow"].sum() == 0
    assert out[(out.zone == "A")]["outflow"].sum() == 0


def test_sudden_jump_is_dropped_as_id_switch_suspect(tmp_path):
    # track 5: A 안에서 5프레임 조용히 있다가, 한 프레임만에 B 안으로 순간이동(occlusion 중 ID 스위칭 흉내)
    # 한 뒤 B 안에 계속 머무른다. 실제 보행 속도로는 불가능한 점프라 이 전환은 세지 않아야 한다.
    rows = [[f, f * 33, 5, 10, 10, 20, 20, 0.9] for f in range(5)]
    rows += [[f, f * 33, 5, 210, 10, 220, 20, 0.9] for f in range(5, 10)]
    csv_path = tmp_path / "det.csv"
    _write_detections(csv_path, rows)

    out = compute_zone_timeseries(str(csv_path), [ZONE_A, ZONE_B], fps=30.0)
    assert out[(out.zone == "B")]["inflow"].sum() == 0
    assert out[(out.zone == "A")]["outflow"].sum() == 0


def test_empty_zone_is_zero_not_missing(tmp_path):
    rows = [[f, f * 33, 1, 10, 10, 20, 20, 0.9] for f in range(5)]
    csv_path = tmp_path / "det.csv"
    _write_detections(csv_path, rows)

    out = compute_zone_timeseries(str(csv_path), [ZONE_A, ZONE_B], fps=30.0)
    b_rows = out[out.zone == "B"]
    assert len(b_rows) > 0
    assert (b_rows["count"] == 0).all()


def test_load_zones_from_json(tmp_path):
    import json

    config = {
        "zones": [
            {"name": "A", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]},
            {"name": "B", "polygon": [[200, 0], [300, 0], [300, 100], [200, 100]]},
        ]
    }
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(config))

    zones = load_zones_from_json(str(path))
    assert [z.name for z in zones] == ["A", "B"]
    assert zones[0].polygon[0] == (0, 0)
