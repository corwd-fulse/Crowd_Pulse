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


def test_angle_near_0_360_boundary_averages_correctly(tmp_path):
    # 프레임마다 오른쪽으로 5px 이동하되 위/아래로 1px씩 번갈아 흔들려서, 프레임별 각도가
    # 0도 바로 양옆(약 348.7도, 11.3도)을 오간다. 이를 "각도끼리" 산술평균하면 (348.7+11.3)/2
    # =180도라는 완전히 틀린 값이 나온다(원형 통계 버그). 벡터로 평균 내면 0도 근처가 맞다.
    base = _textured_frame()
    frames = []
    for i in range(10):
        dy = 1 if i % 2 == 0 else -1
        shifted = np.roll(base, shift=i * 5, axis=1)
        shifted = np.roll(shifted, shift=dy, axis=0)
        frames.append(shifted)
    video_path = tmp_path / "boundary.mp4"
    _write_video(video_path, frames)

    out = compute_zone_motion(str(video_path), [ZONE], fps=10)
    moving = out.dropna(subset=["angle_deg"])
    assert not moving.empty
    mean_angle = moving["angle_deg"].mean()
    # 180도 근처(버그가 났을 때 값)가 아니라 0/360도 근처여야 한다
    assert min(mean_angle, 360 - mean_angle) < 45
