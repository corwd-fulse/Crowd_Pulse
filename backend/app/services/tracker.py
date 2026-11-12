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
from app.services.inference_runtime import check_device, predictor_settings, InferenceFailure

MODEL_NAME = "yolo11m.pt"
IMGSZ = 1920
TRACKER = "bytetrack.yaml"

_model: YOLO | None = None


def _get_model(model_path: Path | None = None) -> YOLO:
    global _model
    if model_path is not None:
        if not Path(model_path).is_file():
            raise InferenceFailure('LOCAL_MODEL_REQUIRED_NO_AUTO_DOWNLOAD')
        return YOLO(str(model_path))  # Fresh tracking state for each Worker job.
    if _model is None:
        _model = YOLO(MODEL_NAME)
    return _model


def track_video(src_path: Path, out_dir: Path, name: str, *, model_path: Path | None = None,
                imgsz: int = IMGSZ, device: str = 'cpu') -> dict:
    """영상에서 사람을 탐지/추적하고 프레임별 인원 수 CSV를 생성한다.

    Returns: 처리 요약 (프레임 수, FPS, 평균/최대 인원 수)
    """
    if type(imgsz) is not int or imgsz <= 0 or imgsz % 32:
        raise ValueError('imgsz must be a positive multiple of 32')
    device = check_device(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_csv = out_dir / f"{name}_frames.csv"
    detections_csv = out_dir / f"{name}_detections.csv"
    count_per_sec_csv = out_dir / f"{name}_count_per_sec.csv"

    cap = cv2.VideoCapture(str(src_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    finally:
        cap.release()
    model = _get_model(model_path)
    execution_settings = {}
    hooks = []
    def verify(predictor):
        execution_settings.update(predictor_settings(predictor, device))
        def verify_input(module, args):
            if args and str(args[0].device) != device:
                raise InferenceFailure('INFERENCE_INPUT_DEVICE_MISMATCH')
        hooks.append(predictor.model.model.register_forward_pre_hook(verify_input))
    model.add_callback('on_predict_start', verify)
    t0 = time.time()

    try:
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
                    imgsz=imgsz,
                    device=device,
                    batch=1,
                    half=False,
                    tracker=TRACKER,
                    verbose=False,
                )
            ):
                if not execution_settings:
                    raise InferenceFailure('INFERENCE_DEVICE_NOT_VERIFIED')
                if str(r.boxes.data.device) != device:
                    raise InferenceFailure('INFERENCE_OUTPUT_DEVICE_MISMATCH')
                t_ms = int(i / fps * 1000)
                b = r.boxes
                ids = b.id.int().tolist() if b.id is not None else [None] * len(b)
                wf.writerow([i, t_ms, len(b)])
                for tid, xyxy, c in zip(ids, b.xyxy.tolist(), b.conf.tolist()):
                    wd.writerow([i, t_ms, tid, *[round(v, 1) for v in xyxy], round(c, 3)])
                frame_count = i + 1
    finally:
        for hook in hooks:
            hook.remove()
        model.callbacks['on_predict_start'].remove(verify)

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
        "execution_settings": {**execution_settings, 'imgsz': imgsz, 'batch': 1, 'half': False,
                               'tracker': TRACKER, 'model_name': Path(model_path or MODEL_NAME).name},
    }
