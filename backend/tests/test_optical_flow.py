import cv2
import numpy as np

from app.services.optical_flow import compute_zone_motion
from app.services.zone_analysis import Zone

ZONE = Zone("A", [(0, 0), (120, 0), (120, 120), (0, 120)])


def _write_video(path, frames, fps=10):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def _textured_frame(size=120):
    # Farneback이 추적할 수 있는 매끄러운 그래디언트 텍스처(순수 노이즈는 약간의 이동에도
    # 전체 프레임 밝기차가 커져 scene-change 휴리스틱을 오탐시킨다).
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (size, size), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (15, 15), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_static_video_has_near_zero_magnitude(tmp_path):
    base = _textured_frame()
    video_path = tmp_path / "static.mp4"
    _write_video(video_path, [base] * 10)

    out = compute_zone_motion(str(video_path), [ZONE], fps=10)
    assert not out.empty
    assert out["magnitude_px"].dropna().mean() < 1.0


def test_shifting_video_has_positive_magnitude_and_consistent_direction(tmp_path):
    base = _textured_frame()
    frames = []
    for i in range(10):
        shifted = np.roll(base, shift=i * 2, axis=1)  # 오른쪽으로 평행 이동
        frames.append(shifted)
    video_path = tmp_path / "moving.mp4"
    _write_video(video_path, frames)

    out = compute_zone_motion(str(video_path), [ZONE], fps=10)
    moving = out.dropna(subset=["magnitude_px"])
    assert not moving.empty
    assert moving["magnitude_px"].mean() > 0.5
