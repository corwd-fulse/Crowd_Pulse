"""밀집 히트맵, 탐지 좌표 그리드 집계 (새 모델 없이 tracker.py 출력 재사용).

CSRNet/DM-Count 같은 별도 밀집도 추정 모델을 쓰지 않고, 이미 갖고 있는 YOLO 탐지 박스의
bottom-center 좌표를 프레임 그리드 셀에 누적해 "어디에 사람이 몰렸는지"를 낸다. 실제 인원수
추정이 아니라 탐지된 사람 수의 공간 분포이며, 카메라 원근 때문에 먼 쪽 셀은 실제보다 밀도가
낮게 보일 수 있다(보정 안 함, 연구용 참고 지표).
"""

from __future__ import annotations

import pandas as pd

GRID_SIZE = 20  # 가로/세로 셀 개수 (스펙 제안값)
WINDOW_S = 10  # 시간 윈도우 (bottleneck.py/queue_analysis.py와 같은 기본값)


def compute_density_heatmap(
    detections_csv: str,
    frame_width: int,
    frame_height: int,
    fps: float,
    grid_size: int = GRID_SIZE,
    window_s: int = WINDOW_S,
) -> pd.DataFrame:
    """반환: columns = [window_start, cell_x, cell_y, count]. count는 그 윈도우 동안
    해당 셀에서 관측된 탐지 박스 수를 윈도우 길이(초)로 나눈 "초당 평균 밀도"다.
    """
    det = pd.read_csv(detections_csv)
    if det.empty:
        return pd.DataFrame(columns=["window_start", "cell_x", "cell_y", "count"])

    cx = (det.x1 + det.x2) / 2
    cy = det.y2
    cell_w = frame_width / grid_size
    cell_h = frame_height / grid_size
    det = det.assign(
        cell_x=(cx // cell_w).clip(0, grid_size - 1).astype(int),
        cell_y=(cy // cell_h).clip(0, grid_size - 1).astype(int),
        sec=det.t_ms // 1000,
    )
    det["window"] = det["sec"] // window_s * window_s

    grouped = det.groupby(["window", "cell_x", "cell_y"]).size().reset_index(name="n")
    grouped["count"] = (grouped["n"] / window_s).round(2)
    return grouped[["window", "cell_x", "cell_y", "count"]].rename(columns={"window": "window_start"})
