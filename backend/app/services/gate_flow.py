"""카메라 한 대를 '문(게이트)'으로 보고, 가상 경계선을 지나간 사람 수를 방향별·초별로 센다.

축제 전체 흐름 지도(venue festival flow)에서 구역 인원 = 시작 인원 + 누적 유입 - 누적 유출 을 계산하는 입력이다.
- axis='x': 화면 세로선(x = line)을 왼쪽으로(minus) / 오른쪽으로(plus) 지나간 수. 가운데 통로 카메라용
- axis='y': 화면 가로선(y = line)을 위로(minus, 카메라에서 멀어짐) / 아래로(plus, 카메라 쪽) 지나간 수. 출입구 카메라용
사람 위치는 박스 하단 중앙(발 위치). 같은 track_id가 선 양쪽에서 각각 HOLD_FRAMES 이상 확인될 때만 통과로 센다
(선 근처에서 흔들리거나 박스가 튀는 것을 통과로 세지 않음). 카메라 간 같은 사람은 이어 붙이지 않는다.
결과는 실제 이동의 하한(추적이 끊기면 못 셈)이며 고유 방문자 수가 아니다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HOLD_FRAMES = 3
MARGIN = 0.01  # 선 양옆 이 비율(화면 크기 대비) 안은 어느 쪽도 아닌 것으로 본다


def gate_counts(detections_csv: Path, axis: str, line: float, size: tuple[int, int], fps: float,
                roi: list | None = None) -> pd.DataFrame:
    """초별 [sec, plus, minus]. line은 화면 비율(0~1). roi(다각형)를 주면 그 안의 발 위치만 쓴다."""
    d = pd.read_csv(detections_csv, usecols=['frame', 'track_id', 'x1', 'y1', 'x2', 'y2']).dropna(subset=['track_id'])
    d['track_id'] = d.track_id.astype(int)
    fx, fy = (d.x1 + d.x2) / 2, d.y2
    if roi is not None:
        from matplotlib.path import Path as MplPath
        inside = MplPath(np.asarray(roi, float)).contains_points(np.column_stack([fx, fy]))
        d, fx, fy = d[inside], fx[inside], fy[inside]
    w, h = size
    pos, span = (fx / w, w) if axis == 'x' else (fy / h, h)
    side = np.where(pos < line - MARGIN, -1, np.where(pos > line + MARGIN, 1, 0))
    d = d.assign(side=side)[side != 0].sort_values(['track_id', 'frame'])
    rows = []
    for tid, g in d.groupby('track_id', sort=False):
        s, f = g.side.to_numpy(), g.frame.to_numpy()
        cur, run, run_side = 0, 0, 0
        for si, fi in zip(s, f):
            run = run + 1 if si == run_side else 1
            run_side = si
            if run >= HOLD_FRAMES:
                if cur and si != cur:
                    rows.append((int(fi / fps), 'plus' if si > cur else 'minus'))
                cur = si
    out = pd.DataFrame(rows, columns=['sec', 'dir'])
    n = int(d.frame.max() / fps) + 1 if len(d) else 0
    res = pd.DataFrame({'sec': range(n)})
    for k in ('plus', 'minus'):
        res[k] = res.sec.map(out[out.dir == k].sec.value_counts()).fillna(0).astype(int)
    return res


def write(job_dir: Path, job_id: str, counts: pd.DataFrame, meta: dict) -> Path:
    p = Path(job_dir) / 'gate_flow.json'
    p.write_text(json.dumps({'job_id': job_id, **meta, 'seconds': int(len(counts)),
                             'plus': counts.plus.tolist(), 'minus': counts.minus.tolist()}, ensure_ascii=False), encoding='utf-8')
    return p


def roi_mid_line(roi_polygon: list, axis: str, size: tuple[int, int]) -> float:
    """ROI 다각형의 가운데(가로 또는 세로)를 화면 비율로. 기본 경계선 위치."""
    p = np.asarray(roi_polygon, float)
    k, n = (0, size[0]) if axis == 'x' else (1, size[1])
    return float((p[:, k].min() + p[:, k].max()) / 2 / n)


def compute_for_job_dir(out_dir: Path, axis: str, line: float | None = None) -> tuple[pd.DataFrame, dict]:
    """분석 출력 폴더(roi_snapshot.json, execution_metadata.json, tracker/video_detections.csv) -> 초별 통과 수."""
    out_dir = Path(out_dir)
    roi = json.loads((out_dir / 'roi_snapshot.json').read_text(encoding='utf-8'))
    size = tuple(roi['video_resolution'])
    fps = float(json.loads((out_dir / 'execution_metadata.json').read_text(encoding='utf-8'))['fps'])
    poly = roi['zones'][0]['polygon']
    line = roi_mid_line(poly, axis, size) if line is None else line
    counts = gate_counts(out_dir / 'tracker' / 'video_detections.csv', axis, line, size, fps, poly)
    return counts, {'axis': axis, 'line': round(line, 4), 'fps': fps, 'roi': roi['zones'][0]['name'],
                    'plus_means': 'right' if axis == 'x' else 'down (toward camera)',
                    'minus_means': 'left' if axis == 'x' else 'up (away from camera)'}
