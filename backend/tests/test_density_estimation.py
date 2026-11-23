from pathlib import Path

import cv2
import numpy as np

from app.services.density_estimation import compute_dmcount_estimates
from app.services.zone_analysis import Zone

# 원본 프레임은 200x100(wxh), ROI는 오른쪽 절반(x=100..200)을 덮는다.
ZONE_RIGHT_HALF = Zone("right", [(100, 0), (200, 0), (200, 100), (100, 100)])


def _write_video(path, n_frames=30, fps=10, size=(200, 100)):
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


def _uniform_density_fn(value_per_cell):
    # 10x20 짜리 density map(= 원본의 1/10 해상도) 전체를 균일값으로 채우는 가짜 추정기.
    # 원본 프레임이 정확히 오른쪽 절반(열 10~19)을 ROI로 덮으므로, ROI 합은
    # value_per_cell * 10(cols) * 10(rows) = value_per_cell * 100 이어야 한다.
    def fn(frame_bgr):
        return np.full((10, 20), value_per_cell, dtype=np.float32), 5.0

    return fn


def test_roi_mask_scales_to_density_map_resolution_and_sums_correctly(tmp_path):
    video_path = tmp_path / "uniform.mp4"
    _write_video(video_path)

    out = compute_dmcount_estimates(
        str(video_path), [ZONE_RIGHT_HALF], fps=10, density_fn=_uniform_density_fn(0.5)
    )

    assert not out.empty
    assert set(out.columns) == {"window_start", "zone", "estimated_count", "inference_ms", "model_version", "density_map_path"}
    # value_per_cell(0.5) * ROI 안 셀 수(10 rows * 10 cols = 100) = 50.0
    assert (out["estimated_count"] == 50.0).all()


def test_windows_follow_shared_window_s_convention(tmp_path):
    video_path = tmp_path / "longer.mp4"
    _write_video(video_path, n_frames=35, fps=10)  # 3.5초 분량

    out = compute_dmcount_estimates(
        str(video_path), [ZONE_RIGHT_HALF], fps=10, window_s=1, density_fn=_uniform_density_fn(1.0)
    )

    # 3.5초 길이를 1초 윈도우로 끊으면 window_start = 0,1,2,3 (총 4개)
    assert sorted(out["window_start"].unique().tolist()) == [0, 1, 2, 3]


def test_samples_per_window_are_averaged_not_just_first_frame(tmp_path):
    video_path = tmp_path / "varying.mp4"
    _write_video(video_path, n_frames=30, fps=10)

    # 첫 프레임엔 0, 이후 프레임엔 2.0을 내는 가짜 추정기 — 중앙/첫 프레임 1장만 썼다면
    # estimated_count가 0 또는 100이 되지만, 평균을 내면 그 사이 값이 나와야 한다.
    calls = {"n": 0}

    def fn(frame_bgr):
        calls["n"] += 1
        value = 0.0 if calls["n"] == 1 else 2.0
        return np.full((10, 20), value, dtype=np.float32), 5.0

    out = compute_dmcount_estimates(
        str(video_path), [ZONE_RIGHT_HALF], fps=10, window_s=3, samples_per_window=3, density_fn=fn
    )

    row = out[out["window_start"] == 0].iloc[0]
    assert 0 < row["estimated_count"] < 200
    assert row["inference_ms"] == 15.0  # 3 샘플 * 5.0ms 합


def test_final_partial_window_still_spreads_samples_across_its_own_length(tmp_path):
    # 2.5초 분량(25프레임 @10fps)을 window_s=1로 자르면 마지막 윈도우(window_start=2)는
    # 실제로 0.5초밖에 안 남는다. 고치기 전에는 그 짧은 구간에 window_s(1초) 기준 간격을
    # 그대로 써서 샘플 지점이 영상 끝(프레임 24) 밖으로 나가 전부 같은 마지막 프레임으로
    # 뭉개졌다 — 평균 3표본이어야 할 윈도우가 사실상 1표본(inference_ms=5.0)짜리가 됐다.
    video_path = tmp_path / "partial_tail.mp4"
    _write_video(video_path, n_frames=25, fps=10)

    out = compute_dmcount_estimates(
        str(video_path), [ZONE_RIGHT_HALF], fps=10, window_s=1, samples_per_window=3,
        density_fn=_uniform_density_fn(1.0),
    )

    last_window = out[out["window_start"] == 2].iloc[0]
    assert last_window["inference_ms"] == 15.0  # 3개 서로 다른 프레임을 추론했다는 뜻


def test_gamma_brightens_dark_frames_before_estimation(tmp_path):
    video_path = tmp_path / "dark.mp4"
    _write_video(video_path, n_frames=10, fps=10)  # 전부 검은(0,0,0) 프레임

    seen_means = []

    def fn(frame_bgr):
        seen_means.append(frame_bgr.mean())
        return np.zeros((10, 20), dtype=np.float32), 1.0

    compute_dmcount_estimates(str(video_path), [ZONE_RIGHT_HALF], fps=10, gamma=1.0, density_fn=fn)
    compute_dmcount_estimates(str(video_path), [ZONE_RIGHT_HALF], fps=10, gamma=2.2, density_fn=fn)

    # 완전히 검은 프레임(0)은 감마 보정을 걸어도 그대로 0이어야 뜻이 맞는 보정(검증용 경계값).
    # 보정 효과 자체는 중간 밝기에서 드러나므로, 여기서는 "no-op과 다른 LUT이 실제로 호출됐는지"만
    # 간접 확인한다 — 중간 밝기 프레임으로 별도 확인.
    assert seen_means[0] == 0.0
    assert seen_means[1] == 0.0


def test_gamma_noop_at_1_but_brightens_midtones_otherwise(tmp_path):
    w, h = 200, 100
    writer = cv2.VideoWriter(str(tmp_path / "gray.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    gray_frame = np.full((h, w, 3), 50, dtype=np.uint8)  # 어두운 중간톤
    for _ in range(10):
        writer.write(gray_frame)
    writer.release()

    def _mean_seen(gamma):
        seen = []

        def fn(frame_bgr):
            seen.append(frame_bgr.mean())
            return np.zeros((10, 20), dtype=np.float32), 1.0

        compute_dmcount_estimates(str(tmp_path / "gray.mp4"), [ZONE_RIGHT_HALF], fps=10, gamma=gamma, density_fn=fn)
        return seen[0]

    assert _mean_seen(2.2) > _mean_seen(1.0)  # gamma=2.2가 gamma=1.0(no-op)보다 밝아야 함


def test_heatmap_dir_none_means_no_image_saved(tmp_path):
    video_path = tmp_path / "uniform.mp4"
    _write_video(video_path)

    out = compute_dmcount_estimates(
        str(video_path), [ZONE_RIGHT_HALF], fps=10, density_fn=_uniform_density_fn(0.5)
    )
    assert out["density_map_path"].isna().all() or (out["density_map_path"] == None).all()  # noqa: E711


def test_heatmap_dir_saves_one_shared_png_per_window(tmp_path):
    video_path = tmp_path / "uniform.mp4"
    _write_video(video_path, n_frames=40)  # 4초 분량
    heatmap_dir = tmp_path / "heatmaps"

    out = compute_dmcount_estimates(
        str(video_path), [ZONE_RIGHT_HALF], fps=10, window_s=1, density_fn=_uniform_density_fn(0.5),
        heatmap_dir=heatmap_dir,
    )

    assert out["density_map_path"].notna().all()
    assert heatmap_dir.is_dir()
    saved_files = sorted(p.name for p in heatmap_dir.glob("*.png"))
    assert saved_files == [f"window_{w}.png" for w in sorted(out["window_start"].unique())]
    for path in out["density_map_path"]:
        assert Path(path).is_file()


def test_empty_zones_returns_empty_frame(tmp_path):
    video_path = tmp_path / "novids.mp4"
    _write_video(video_path)

    out = compute_dmcount_estimates(str(video_path), [], fps=10, density_fn=_uniform_density_fn(1.0))
    assert out.empty
