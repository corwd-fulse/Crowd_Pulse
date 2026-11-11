"""YOLO + ByteTrack 기반 인원 탐지/추적.

Colab 실험(MOT20 idf1/mota 비교)에서 가장 정확도가 높았던 조합을 사용:
yolo11m.pt + imgsz=1920 + tracker=bytetrack.yaml
"""

from pathlib import Path
import csv
import time

import cv2
import pandas as pd
from ultralytics import YOLO

MODEL_NAME = "yolo11m.pt"
IMGSZ = 1920
TRACKER = "bytetrack.yaml"

_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_NAME)
    return _model


def track_video(src_path: Path, out_dir: Path, name: str) -> dict:
    """영상에서 사람을 탐지/추적하고 프레임별 인원 수 CSV를 생성한다.

    Returns: 처리 요약 (프레임 수, FPS, 평균/최대 인원 수)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_csv = out_dir / f"{name}_frames.csv"
    detections_csv = out_dir / f"{name}_detections.csv"
    count_per_sec_csv = out_dir / f"{name}_count_per_sec.csv"

    fps = cv2.VideoCapture(str(src_path)).get(cv2.CAP_PROP_FPS) or 30.0
    model = _get_model()
    t0 = time.time()

    with open(frames_csv, "w", newline="") as ff, open(detections_csv, "w", newline="") as fd:
        wf, wd = csv.writer(ff), csv.writer(fd)
        wf.writerow(["frame", "t_ms", "count"])
        wd.writerow(["frame", "t_ms", "track_id", "x1", "y1", "x2", "y2", "conf"])

        frame_count = 0
        for i, r in enumerate(
            model.track(
                str(src_path),
                stream=True,
                classes=[0],
                imgsz=IMGSZ,
                tracker=TRACKER,
                verbose=False,
            )
        ):
            t_ms = int(i / fps * 1000)
            b = r.boxes
            ids = b.id.int().tolist() if b.id is not None else [None] * len(b)
            wf.writerow([i, t_ms, len(b)])
            for tid, xyxy, c in zip(ids, b.xyxy.tolist(), b.conf.tolist()):
                wd.writerow([i, t_ms, tid, *[round(v, 1) for v in xyxy], round(c, 3)])
            frame_count = i + 1

    f = pd.read_csv(frames_csv)
    f["sec"] = f.t_ms // 1000
    f.groupby("sec")["count"].mean().round(1).to_csv(count_per_sec_csv)

    elapsed = time.time() - t0
    return {
        "frame_count": frame_count,
        "fps_processed": round(frame_count / elapsed, 1) if elapsed > 0 else 0,
        "mean_count": round(float(f["count"].mean()), 1) if frame_count else 0,
        "max_count": int(f["count"].max()) if frame_count else 0,
        "frames_csv": str(frames_csv),
        "detections_csv": str(detections_csv),
        "count_per_sec_csv": str(count_per_sec_csv),
    }
