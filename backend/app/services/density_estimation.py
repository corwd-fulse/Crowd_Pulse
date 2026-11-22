"""DM-Count 밀도맵 기반 구역별 인원 추정 (FR-DEN-01/02, P2).

YOLO+ByteTrack(tracker.py)/Optical Flow(optical_flow.py)와는 입력 영상 프레임만 공유하는
완전 독립 파이프라인이다. 개체를 하나도 식별하지 않고, 프레임 전체를 모델에 통과시켜 나온
밀도맵(각 픽셀이 "그 위치에 분배된 인원수")을 ROI 영역만 합산해 구역별 추정치를 낸다.

같은 영상의 YOLO 구역 카운트(zone_analysis.compute_zone_timeseries)와 비교할 목적이므로,
비교 기준을 맞추기 위해 다른 윈도우 지표(bottleneck.py/queue_analysis.py)와 동일한
WINDOW_S=10초 단위로 집계한다. YOLO는 그 시점 ROI 안의 "고유 인원 수"이고 이 모듈의 값은
"그 순간의 밀도 추정치"다 — ByteTrack 누적 유입/유출과는 비교 대상이 아니다.

shortcut: 윈도우 전체가 아니라 윈도우당 몇 장(samples_per_window)만 균등 간격으로 추론해서
평균 낸다. 실제 영상(30초 클립)으로 중앙 프레임 1장만 쓰던 첫 버전을 돌려보니 10s/20s
윈도우에서 YOLO 대비 절반 이하로 떨어지는 윈도우가 있었다 — 그 1장이 하필 사람이 적게
찍힌 순간이었을 가능성이 커서 여러 장 평균으로 바꿨다. DM-Count 추론이 이 파이프라인에서
제일 비싼 연산이라 매 프레임 돌리지는 않고, density map 이미지 자체도 저장하지 않는다
(ROI 합산 스칼라만 남김 — 디스크 비용 삭제).

QNRF/ShanghaiTech Part B pretrained 체크포인트 비교 실험(야간·우천 CCTV 클립)에서 Part B는
프레임 전체 density 합이 정확히 0.0으로 붕괴했다 — 레포 예시의 밝은 주간 이미지에는 정상
작동했으므로 체크포인트 문제가 아니라 저조도 도메인 불일치였다. gamma 보정(기본 비활성,
gamma=1.0)은 그 가설에 대한 대응 — 모델에 넣기 전에 어두운 영역만 들어올려 두 모델이 학습
때 본 것과 비슷한 밝기 분포로 맞춰준다.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

from app.services.inference_runtime import InferenceFailure, check_device
from app.services.zone_analysis import Zone

WINDOW_S = 10  # bottleneck.py/queue_analysis.py와 같은 윈도우

DensityFn = Callable[[np.ndarray], tuple[np.ndarray, float]]  # BGR 프레임 -> (density map, 추론ms)

_model_cache: dict[str, object] = {}


def _load_model(model_path: Path, device: str):
    key = f"{model_path}:{device}"
    if key not in _model_cache:
        if not model_path.is_file():
            raise InferenceFailure('LOCAL_MODEL_REQUIRED_NO_AUTO_DOWNLOAD')
        import torch

        from app.services.dmcount_model import vgg19

        model = vgg19()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        _model_cache[key] = model
    return _model_cache[key]


def _estimator_from_model(model_path: Path, device: str) -> DensityFn:
    import torch

    model = _load_model(model_path, device)

    def estimate(frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inp = transforms.ToTensor()(img).unsqueeze(0).to(device)
        start = time.perf_counter()
        with torch.no_grad():
            outputs, _ = model(inp)
        inference_ms = (time.perf_counter() - start) * 1000
        return outputs[0, 0].cpu().numpy(), inference_ms

    return estimate


_gamma_lut_cache: dict[float, np.ndarray] = {}


def _apply_gamma(frame_bgr: np.ndarray, gamma: float) -> np.ndarray:
    """어두운 픽셀을 들어올리는 감마 보정. gamma=1.0은 입력을 그대로 반환(no-op)."""
    if gamma == 1.0:
        return frame_bgr
    if gamma not in _gamma_lut_cache:
        levels = np.arange(256, dtype=np.float32) / 255.0
        _gamma_lut_cache[gamma] = (levels ** (1.0 / gamma) * 255).clip(0, 255).astype(np.uint8)
    return cv2.LUT(frame_bgr, _gamma_lut_cache[gamma])


def _save_heatmap(density_map: np.ndarray, path: Path) -> None:
    """density map을 보기용 컬러 히트맵 PNG로 저장한다(모델 원본 해상도 그대로, 리사이즈는
    표시용으로만 — 집계에 쓰는 합산값과는 무관하다)."""
    vis = density_map.astype(np.float32)
    vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-6)
    vis = (vis * 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.applyColorMap(vis, cv2.COLORMAP_JET))


def _zone_mask_at(zone: Zone, frame_shape: tuple[int, int], map_shape: tuple[int, int]) -> np.ndarray:
    """frame_shape(원본 h,w) 좌표의 ROI 폴리곤을 map_shape(density map h,w) 격자로 스케일해 마스크를 만든다.

    density map 자체는 리사이즈하지 않는다(밀도 합이 보존되도록) — 대신 ROI 좌표 쪽을
    density map 해상도에 맞춰 축소해서, 모델이 낸 해상도 그대로 합산한다.
    """
    frame_h, frame_w = frame_shape
    map_h, map_w = map_shape
    sx, sy = map_w / frame_w, map_h / frame_h
    pts = np.array([(x * sx, y * sy) for x, y in zone.polygon], dtype=np.int32)
    mask = np.zeros(map_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def compute_dmcount_estimates(
    video_path: str,
    zones: list[Zone],
    *,
    fps: float | None = None,
    model_path: Path | str = Path("models/dmcount_qnrf.pth"),
    device: str = "cpu",
    model_version: str = "dmcount_qnrf",
    window_s: int = WINDOW_S,
    samples_per_window: int = 3,
    gamma: float = 1.0,
    heatmap_dir: Path | str | None = None,
    density_fn: DensityFn | None = None,
) -> pd.DataFrame:
    """반환: columns = [window_start, zone, estimated_count, inference_ms, model_version,
    density_map_path].

    heatmap_dir: 지정하면(기본값 None=저장 안 함) 윈도우당 1장(마지막으로 추론한 샘플의
    전체 프레임 density map)을 컬러 히트맵 PNG로 그 디렉터리에 저장하고, 그 윈도우의 모든
    zone 행에 같은 파일 경로를 density_map_path로 채운다 — density map은 zone별이 아니라
    프레임 전체에서 한 번만 나오므로 같은 윈도우의 zone들은 같은 이미지를 공유한다.
    생략하면 모든 행의 density_map_path는 None(디스크 비용 없음, 기존 동작과 동일).

    estimated_count: 윈도우 안에서 균등 간격으로 뽑은 samples_per_window개 프레임 각각의
    ROI 합산값을 평균한 값. inference_ms: 그 샘플들 추론 시간의 합(그 윈도우를 처리하는 데
    실제로 든 총 비용).
    gamma: 1.0(기본, 보정 없음)보다 크면 어두운 픽셀을 들어올리는 감마 보정을 모델 입력
    직전에 적용한다. 저조도 영상에서 pretrained 모델이 장면을 아예 못 읽는 문제(모듈
    docstring 참고)에 대한 대응.
    density_fn: 테스트에서 실제 가중치 없이 ROI 스케일링/집계/평균 로직만 검증하려 할 때
    주입하는 가짜 추정 함수. 생략하면(기본값) model_path의 실제 DM-Count 가중치로 추론한다.
    gamma 보정은 density_fn을 쓸 때도 똑같이 적용된다(추정 함수가 아니라 프레임 읽기
    직후에 적용하므로).
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    if samples_per_window <= 0:
        raise ValueError("samples_per_window must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    device = check_device(device)
    estimate = density_fn or _estimator_from_model(Path(model_path), device)
    if heatmap_dir is not None:
        heatmap_dir = Path(heatmap_dir)
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    try:
        fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        columns = ["window_start", "zone", "estimated_count", "inference_ms", "model_version", "density_map_path"]
        if total_frames <= 0 or not zones:
            return pd.DataFrame(columns=columns)

        duration_s = total_frames / fps
        rows = []
        window_start = 0.0
        while window_start < duration_s:
            # 윈도우 경계에 걸치지 않도록 내부 균등 간격 지점을 쓴다(0초/10초 자체가 아니라
            # 그 사이 (1/(n+1), 2/(n+1), ...) 지점). 영상 끝에 걸린 마지막 윈도우는
            # window_s보다 실제로 더 짧으므로, 그 실제 남은 길이(window_len) 안에서
            # 간격을 다시 분배한다 — 그러지 않으면 샘플 지점들이 전부 영상 끝 프레임으로
            # min(..., total_frames-1) 클리핑되어 사실상 같은 프레임 1장만 쓰게 된다
            # (실제 30초 클립 테스트에서 마지막 윈도우가 3배 평균이 아니라 1회분으로
            # 나온 원인). 짧은 윈도우/긴 샘플 수라 그래도 같은 프레임 인덱스가 겹치면
            # 중복 추론을 피하려고 set으로 합친다(그만큼 평균 표본 수가 준다).
            window_len = min(window_s, duration_s - window_start)
            offsets = [window_len * (i + 1) / (samples_per_window + 1) for i in range(samples_per_window)]
            frame_indices = sorted({min(int((window_start + off) * fps), total_frames - 1) for off in offsets})

            counts_per_zone: dict[str, list[float]] = {z.name: [] for z in zones}
            total_inference_ms = 0.0
            last_density_map = None
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = _apply_gamma(frame, gamma)
                density_map, inference_ms = estimate(frame)
                total_inference_ms += inference_ms
                last_density_map = density_map
                for zone in zones:
                    mask = _zone_mask_at(zone, (frame_h, frame_w), density_map.shape)
                    counts_per_zone[zone.name].append(float(density_map[mask == 1].sum()))

            heatmap_path = None
            if heatmap_dir is not None and last_density_map is not None:
                heatmap_path = heatmap_dir / f"window_{int(window_start)}.png"
                _save_heatmap(last_density_map, heatmap_path)
                heatmap_path = str(heatmap_path)

            for zone in zones:
                samples = counts_per_zone[zone.name]
                if not samples:
                    continue
                rows.append({
                    "window_start": int(window_start),
                    "zone": zone.name,
                    "estimated_count": round(sum(samples) / len(samples), 2),
                    "inference_ms": round(total_inference_ms, 1),
                    "model_version": model_version,
                    "density_map_path": heatmap_path,
                })
            window_start += window_s
        return pd.DataFrame(rows, columns=columns)
    finally:
        cap.release()
