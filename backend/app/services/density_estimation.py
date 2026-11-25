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

import hashlib
import json
import os
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


def _load_model(model_path: Path, device: str, half: bool = False):
    key = f"{model_path}:{device}:{'fp16' if half else 'fp32'}"
    if key not in _model_cache:
        if not model_path.is_file():
            raise InferenceFailure('LOCAL_MODEL_REQUIRED_NO_AUTO_DOWNLOAD')
        import torch

        from app.services.dmcount_model import vgg19

        model = vgg19()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        if half:
            model.half()
        model.eval()
        _model_cache[key] = model
    return _model_cache[key]


# DM-Count 학습 코드(upstream datasets/crowd.py)와 같은 ImageNet 정규화. 이게 빠지면 입력
# 분포가 학습 때와 달라져 density 합이 크게 줄어든다(야간 CCTV 실측: YOLO 대비 6% → 37%).
_TO_INPUT = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _estimator_from_model(model_path: Path, device: str, input_scale: float = 1.0, half: bool = False) -> DensityFn:
    import torch

    model = _load_model(model_path, device, half)

    def estimate(frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        # input_scale>1: 멀리 작게 찍힌 사람을 QNRF 학습 때 크기에 가깝게 키운다. density map
        # 합은 인원수라 해상도가 바뀌어도 그대로 쓰고, ROI 마스크는 map 크기에 맞춰 스케일된다.
        if input_scale != 1.0:
            frame_bgr = cv2.resize(frame_bgr, None, fx=input_scale, fy=input_scale, interpolation=cv2.INTER_CUBIC)
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inp = _TO_INPUT(img).unsqueeze(0).to(device)
        if half:
            inp = inp.half()
        start = time.perf_counter()
        with torch.no_grad():
            outputs, _ = model(inp)
        inference_ms = (time.perf_counter() - start) * 1000
        # fp16이어도 density map은 float32로 돌려준다 — ROI 합산(인원수)에서 fp16 누적 오차가
        # 쌓이지 않도록 합산은 항상 float32로 한다.
        return outputs[0, 0].float().cpu().numpy(), inference_ms

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


# YOLO 박스를 사방으로 이만큼(박스 폭/높이 비율) 넓혀서, 그 안의 density는 YOLO가 이미 센
# 사람으로 본다 — 머리가 박스 위끝에 걸쳐 밀도가 박스 밖으로 번지는 것을 흡수하는 여유.
RESIDUAL_BOX_PAD = 0.10


def _detections_lookup(detections_csv: Path):
    """tracker.py의 video_detections.csv를 프레임 번호로 정렬해 두고, 프레임 하나의 박스
    (N,4: x1,y1,x2,y2)를 이진 탐색으로 꺼내는 함수를 돌려준다(긴 영상도 dict 없이 빠르게)."""
    det = pd.read_csv(detections_csv, usecols=["frame", "x1", "y1", "x2", "y2"])
    det = det.sort_values("frame", kind="stable")
    frames = det["frame"].to_numpy()
    xyxy = det[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)

    def boxes_at(frame_idx: int) -> np.ndarray:
        lo, hi = np.searchsorted(frames, frame_idx, "left"), np.searchsorted(frames, frame_idx, "right")
        return xyxy[lo:hi]

    return boxes_at


def _uncovered_mask(boxes: np.ndarray, frame_shape: tuple[int, int], map_shape: tuple[int, int]) -> np.ndarray:
    """density map 격자에서 어느 YOLO 박스(RESIDUAL_BOX_PAD만큼 넓힘)에도 덮이지 않은 칸 = True."""
    frame_h, frame_w = frame_shape
    map_h, map_w = map_shape
    sx, sy = map_w / frame_w, map_h / frame_h
    covered = np.zeros(map_shape, dtype=bool)
    for x1, y1, x2, y2 in boxes:
        px, py = (x2 - x1) * RESIDUAL_BOX_PAD, (y2 - y1) * RESIDUAL_BOX_PAD
        r0, r1 = max(0, int((y1 - py) * sy)), min(map_h, int(np.ceil((y2 + py) * sy)) + 1)
        c0, c1 = max(0, int((x1 - px) * sx)), min(map_w, int(np.ceil((x2 + px) * sx)) + 1)
        covered[r0:r1, c0:c1] = True
    return ~covered


DMCOUNT_CHECKPOINT_VERSION = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _dmcount_fingerprint(**payload) -> str:
    """DM-Count 체크포인트가 같은 (영상,구역,설정,가중치)에서 만든 것인지 확인하는 서명.

    optical_flow.py의 checkpoint 서명과 달리 영상 *경로*와 mtime은 넣지 않는다 — GPU 서버에서
    끊긴 계산을 다른 컴퓨터(다른 GPU)로 옮겨 이어갈 수 있어야 하기 때문이다. 대신 영상 크기·
    프레임 수·fps·해상도와 가중치 파일 sha256으로 "같은 입력"을 판단한다.
    """
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class _DmcountCheckpoint:
    """윈도우 하나가 끝날 때마다 그때까지의 행과 다음 window_start를 JSON 하나로 원자적으로
    교체 저장한다(tmp에 쓰고 os.replace). 행의 히트맵은 절대경로가 아니라 파일 이름만 저장해서,
    작업 디렉터리를 다른 컴퓨터로 복사해도 그대로 이어갈 수 있다."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self, fingerprint: str, heatmap_dir: Path | None,
             has_residual: bool = False) -> tuple[float, list[dict], dict[str, int]] | None:
        if not self.path.is_file():
            return None
        try:
            state = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            print(f'dmcount checkpoint unreadable ({exc!r}); starting DM-Count from the beginning', flush=True)
            return None
        if state.get('has_residual') and not has_residual:
            # 조용히 처음부터 다시 돌리면 GPU 시간을 통째로 버리므로 멈추고 알려준다.
            raise ValueError(
                f'{self.path} was computed with the YOLO residual (tracker detections CSV); to continue it, '
                'provide the same tracker/video_detections.csv (e.g. copy it next to extended/ on this machine), '
                'or delete the checkpoint to restart DM-Count without the residual')
        if state.get('version') != DMCOUNT_CHECKPOINT_VERSION or state.get('fingerprint') != fingerprint:
            print('dmcount checkpoint does not match this video/zones/settings/weights; '
                  'starting DM-Count from the beginning', flush=True)
            return None
        rows = state.get('rows', [])
        for row in rows:
            name = row.get('density_map_path')
            if name and (heatmap_dir is None or not (heatmap_dir / name).is_file()):
                print(f'dmcount checkpoint heatmap {name} missing; starting DM-Count from the beginning', flush=True)
                return None
        return float(state['next_window_start']), rows, dict(state.get('windows_by_precision', {}))

    def save(self, fingerprint: str, next_window_start: float, rows: list[dict],
             windows_by_precision: dict[str, int], has_residual: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + '.tmp')
        tmp.write_text(json.dumps({
            'version': DMCOUNT_CHECKPOINT_VERSION, 'fingerprint': fingerprint, 'has_residual': has_residual,
            'next_window_start': next_window_start, 'rows': rows,
            'windows_by_precision': windows_by_precision,
        }), encoding='utf-8')
        os.replace(tmp, self.path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


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
    input_scale: float = 1.0,
    heatmap_dir: Path | str | None = None,
    density_fn: DensityFn | None = None,
    checkpoint_path: Path | str | None = None,
    half: bool = False,
    detections_csv: Path | str | None = None,
) -> pd.DataFrame:
    """반환: columns = [window_start, zone, estimated_count, inference_ms, model_version,
    density_map_path] (+ yolo_residual_count, detections_csv를 준 경우).

    detections_csv: YOLO+ByteTrack 결과(tracker.py의 video_detections.csv, frame은 영상 프레임
    번호)를 주면, 같은 샘플 프레임의 사람 박스(사방 RESIDUAL_BOX_PAD만큼 넓힘) 바깥에 남은
    density 합을 zone별로 구해 yolo_residual_count에 넣는다 — "YOLO가 못 잡은 곳에서 DM-Count가
    본 인원". YOLO 인원 + 이 값이 YOLO+DM-Count 보완 추정이다. 모델을 더 돌리지 않고 이미 구한
    density map을 나눠 더할 뿐이라 비용은 프레임당 수 ms다. 주의: 꽃·간판 같은 무늬를 DM-Count가
    사람으로 보는 오탐도 박스 밖에 있으면 그대로 더해진다(59분 낮 영상 실측, 참고 지표로만 쓴다).

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
    input_scale: 모델 입력 전에 프레임을 확대/축소할 배율(기본 1.0). 사람이 작게 찍힌
    한산한 장면에서 과소 추정을 줄인다. density_fn을 주입하면 적용되지 않는다(실제 모델
    추정 함수 안에서 적용되므로).
    density_fn: 테스트에서 실제 가중치 없이 ROI 스케일링/집계/평균 로직만 검증하려 할 때
    주입하는 가짜 추정 함수. 생략하면(기본값) model_path의 실제 DM-Count 가중치로 추론한다.
    gamma 보정은 density_fn을 쓸 때도 똑같이 적용된다(추정 함수가 아니라 프레임 읽기
    직후에 적용하므로).
    checkpoint_path: 지정하면 윈도우 하나가 끝날 때마다 진행 상태를 이 JSON에 저장하고, 같은
    서명(_dmcount_fingerprint)의 체크포인트가 이미 있으면 그 다음 윈도우부터 이어서 계산한다.
    서명에 영상 경로가 없으므로 작업 디렉터리를 다른 컴퓨터로 옮겨서(다른 GPU/device로)
    이어가도 된다. 정상 종료 시 체크포인트 파일은 삭제한다. 생략하면 기존 동작과 동일.
    half: True면 모델 가중치·입력을 fp16으로 바꿔 Volta 이후(텐서코어) GPU에서 메모리를 약
    절반으로 줄인다(opt-in). CUDA에서만 허용한다. density map 합산은 float32로 하므로 인원수
    오차는 작을 것으로 예상하지만(VGG 계열 추론) 실제 가중치로는 아직 실측 전이다.
    주의: Pascal(GTX 10xx)에서는 역효과다 — GTX 1060 3GB, 2880x1620 입력 실측에서 fp16이
    3.5s/장·3.4 GiB(3GB 초과분이 시스템 메모리로 넘침), fp32가 0.94s/장·2.36 GiB였다. Pascal은
    fp16 연산이 느리고 cuDNN fp16 커널도 비효율적이라, 그 세대에서는 fp32를 쓴다.
    정밀도는 서명에 넣지 않는다 — fp32로 하다 끊긴 계산을 fp16(다른 GPU)으로 이어갈 수 있게
    하려는 것이고, 대신 어느 정밀도로 몇 개 윈도우를 계산했는지 반환 DataFrame의
    attrs['windows_by_precision']에 남긴다(예: {'fp32': 120, 'fp16': 234}).
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    if samples_per_window <= 0:
        raise ValueError("samples_per_window must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if input_scale <= 0:
        raise ValueError("input_scale must be positive")
    device = check_device(device)
    if half and device == 'cpu' and density_fn is None:
        raise ValueError("half (fp16) DM-Count inference requires a CUDA device")
    precision = 'fp16' if half else 'fp32'
    estimate = density_fn or _estimator_from_model(Path(model_path), device, input_scale, half)
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
        if detections_csv is not None:
            columns.append("yolo_residual_count")
        if total_frames <= 0 or not zones:
            return pd.DataFrame(columns=columns)
        boxes_at = _detections_lookup(Path(detections_csv)) if detections_csv is not None else None

        duration_s = total_frames / fps
        rows: list[dict] = []
        window_start = 0.0
        windows_by_precision: dict[str, int] = {}
        checkpoint = fingerprint = None
        if checkpoint_path is not None:
            checkpoint = _DmcountCheckpoint(Path(checkpoint_path))
            fingerprint = _dmcount_fingerprint(
                video_size=os.path.getsize(video_path), total_frames=total_frames, fps=fps,
                frame_size=(frame_w, frame_h), zones=[(z.name, z.polygon) for z in zones],
                model_version=model_version, window_s=window_s, samples_per_window=samples_per_window,
                gamma=gamma, input_scale=input_scale, heatmaps=heatmap_dir is not None,
                weights='injected' if density_fn is not None else _file_sha256(Path(model_path)),
                # 잔여(박스 밖) 계산 여부가 다르면 행 구성이 달라지므로 서명에 넣는다. 없을 때는
                # 키 자체를 넣지 않아 기존 체크포인트와 서명이 그대로 맞는다.
                **({'detections': _file_sha256(Path(detections_csv))} if detections_csv is not None else {}),
            )
            restored = checkpoint.load(fingerprint, heatmap_dir, has_residual=boxes_at is not None)
            if restored is not None:
                window_start, rows, windows_by_precision = restored
                print(f'dmcount: resuming from checkpoint at {window_start:g}s / {duration_s:.0f}s '
                      f'({len(rows)} rows restored, so far {windows_by_precision}; continuing in {precision})',
                      flush=True)
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
            residual_per_zone: dict[str, list[float]] = {z.name: [] for z in zones}
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
                uncovered = None
                if boxes_at is not None:
                    uncovered = _uncovered_mask(boxes_at(frame_idx), (frame_h, frame_w), density_map.shape)
                for zone in zones:
                    mask = _zone_mask_at(zone, (frame_h, frame_w), density_map.shape) == 1
                    counts_per_zone[zone.name].append(float(density_map[mask].sum()))
                    if uncovered is not None:
                        residual_per_zone[zone.name].append(float(density_map[mask & uncovered].sum()))

            # 행에는 히트맵 파일 이름만 넣고, 절대경로는 반환 직전에 붙인다(체크포인트 이식성).
            heatmap_name = None
            if heatmap_dir is not None and last_density_map is not None:
                heatmap_name = f"window_{int(window_start)}.png"
                _save_heatmap(last_density_map, heatmap_dir / heatmap_name)

            for zone in zones:
                samples = counts_per_zone[zone.name]
                if not samples:
                    continue
                row = {
                    "window_start": int(window_start),
                    "zone": zone.name,
                    "estimated_count": round(sum(samples) / len(samples), 2),
                    "inference_ms": round(total_inference_ms, 1),
                    "model_version": model_version,
                    "density_map_path": heatmap_name,
                }
                if boxes_at is not None:
                    residual = residual_per_zone[zone.name]
                    row["yolo_residual_count"] = round(sum(residual) / len(residual), 2)
                rows.append(row)
            window_start += window_s
            windows_by_precision[precision] = windows_by_precision.get(precision, 0) + 1
            if checkpoint is not None:
                checkpoint.save(fingerprint, window_start, rows, windows_by_precision,
                                has_residual=boxes_at is not None)
        if heatmap_dir is not None:
            for row in rows:
                if row["density_map_path"]:
                    row["density_map_path"] = str(heatmap_dir / row["density_map_path"])
        result = pd.DataFrame(rows, columns=columns)
        result.attrs["windows_by_precision"] = windows_by_precision
        if checkpoint is not None:
            checkpoint.clear()
        return result
    finally:
        cap.release()
