"""Optical Flow 기반 구역 평균 이동량 계산 (FR-ZON-03).

Farneback 알고리즘으로 연속 프레임 간 이동 벡터를 구하고, 구역 마스크 안
벡터를 평균해 angle_deg/magnitude_px/mean_motion(정규화)으로 요약한다.
m/s 로는 절대 환산하지 않는다(카메라 보정 전 픽셀 단위 보조 지표).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from .zone_analysis import Zone

MIN_VECTORS = 25  # 구역이 너무 작아 벡터 수 부족으로 보는 임계값 (shortcut: 임의값, 실footage로 재조정 필요)
SCENE_CHANGE_DIFF = 40.0  # shortcut: 프레임 평균 밝기 변화가 이 값을 넘으면 장면전환/흔들림으로 간주한 휴리스틱

ProgressCallback = Callable[[int, int], None]


def _zone_mask(zone: Zone, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array(zone.polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def _checkpoint_fingerprint(video_path: str, zones: list[Zone], fps: float, frame_stride: int,
                             flow_method: str) -> str:
    """같은 (영상,구역,설정)에서 만든 checkpoint인지 확인하는 서명.

    영상 파일이 바뀌었는데 예전 checkpoint를 그대로 이어서 계산하면 조용히 틀린 결과를
    내므로, 재개 전에 반드시 이 서명이 일치하는지 검사한다(일치하지 않으면 처음부터 다시).
    """
    st = os.stat(video_path)
    payload = dict(
        video_path=str(video_path), video_size=st.st_size, video_mtime=st.st_mtime,
        zones=[(z.name, z.polygon) for z in zones], fps=fps, frame_stride=frame_stride,
        flow_method=flow_method,
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def compute_checkpoint_fingerprint(video_path: str, zones: list[Zone], fps: float, frame_stride: int,
                                    flow_method: str) -> str:
    """worker.py 등 바깥 호출자가 compute_zone_motion()을 돌리지 않고도, 디스크의 checkpoint가
    "지금 돌리려는 (영상,구역,설정)"과 같은 것인지 미리 확인할 때 쓰는 공개 래퍼."""
    return _checkpoint_fingerprint(video_path, zones, fps, frame_stride, flow_method)


@dataclass(frozen=True)
class CheckpointStatus:
    """checkpoint 파일 하나를 열어본 결과. valid=False면 frame_idx는 항상 None이고,
    reason이 재개하지 않은(또는 못한) 구체적 이유를 담는다 — "그냥 조용히 처음부터 다시
    돌았다"로 끝나지 않도록, 호출자가 reason을 로그/실패사유로 남길 수 있게 한다."""
    valid: bool
    reason: str | None
    frame_idx: int | None


class _Checkpoint:
    """Optical Flow 프레임 루프의 중간 상태를 디스크에 저장/복원한다.

    형식: 매직바이트 + sha256 체크섬(32바이트) + pickle 페이로드를 **한 파일**에 순서대로
    담고, 임시 파일에 쓴 뒤 fsync 후 os.replace()로 "한 번만" 교체한다. 예전에는 상태(json)와
    직전 프레임(npy)을 각각 따로 교체해서, 그 둘 사이에 프로세스가 죽으면 상태와 프레임이
    서로 안 맞는 checkpoint가 남을 수 있었다(한쪽만 새 걸로 바뀜) — 단일 파일 + 단일 교체로
    이 창을 없앤다. 교체 전 파일 자체가 쓰이다 중단되더라도(디스크 꽉 찼거나 kill) 임시
    파일만 깨지고, 이전에 성공적으로 저장된 checkpoint(`self.path`)는 그대로 남는다 — 로드할
    때 체크섬을 검증해서, 혹시라도 깨진 채로 보이면 "손상됨"으로 명확히 보고하고 무시한다.

    범위: 이 checkpoint는 "같은 job 실행(같은 worker 프로세스)" 안에서 optical flow 단계가
    중간에 예외로 끊겼을 때, 처음부터 다시 돌리지 않고 이어서 계산하기 위한 용도로 시작했다.
    worker.py가 서버 재시작 후 재개(cross-restart resume)에도 이 파일을 재사용하지만, 그
    경우의 seek 정확도·job 재개 여부 결정은 worker.py 쪽 로직(resume_manifest 등)이 책임지고,
    이 클래스 자체는 "파일 하나를 안전하게 저장/복원"하는 것만 책임진다.
    """

    _MAGIC = b'CPOFCKPT2'
    _DIGEST_LEN = hashlib.sha256().digest_size

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self, fingerprint: str) -> tuple[tuple[int, dict, dict, np.ndarray] | None, CheckpointStatus]:
        if not self.path.exists():
            return None, CheckpointStatus(False, None, None)  # 그냥 처음 실행, 이상 없음
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
            return None, CheckpointStatus(False, f'checkpoint read failed: {exc}', None)
        if len(raw) < len(self._MAGIC) + self._DIGEST_LEN or not raw.startswith(self._MAGIC):
            return None, CheckpointStatus(False, 'checkpoint truncated or bad header', None)
        digest = raw[len(self._MAGIC):len(self._MAGIC) + self._DIGEST_LEN]
        payload = raw[len(self._MAGIC) + self._DIGEST_LEN:]
        if hashlib.sha256(payload).digest() != digest:
            return None, CheckpointStatus(False, 'checkpoint checksum mismatch (corrupted)', None)
        try:
            state = pickle.loads(payload)
        except Exception as exc:  # noqa: BLE001 — 어떤 식으로 깨졌든 "손상됨"으로 통일 처리
            return None, CheckpointStatus(False, f'checkpoint payload unreadable: {exc!r}', None)
        if not isinstance(state, dict) or not {'fingerprint', 'frame_idx', 'acc', 'last_values', 'prev_gray'} <= state.keys():
            return None, CheckpointStatus(False, 'checkpoint missing required fields', None)
        if state['fingerprint'] != fingerprint:
            return None, CheckpointStatus(
                False, 'checkpoint fingerprint mismatch (different video/zones/settings)', None)
        return (state['frame_idx'], state['acc'], state['last_values'], state['prev_gray']), \
            CheckpointStatus(True, None, state['frame_idx'])

    def save(self, fingerprint: str, frame_idx: int, acc: dict, last_values: dict, prev_gray: np.ndarray) -> None:
        payload = pickle.dumps(
            dict(fingerprint=fingerprint, frame_idx=frame_idx, acc=dict(acc), last_values=dict(last_values),
                prev_gray=prev_gray),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        digest = hashlib.sha256(payload).digest()
        tmp = self.path.with_name(f'{self.path.name}.tmp{os.getpid()}')
        try:
            with open(tmp, 'wb') as f:
                f.write(self._MAGIC)
                f.write(digest)
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)  # 단일 원자적 교체 — 이 한 줄 전까지는 기존 checkpoint 그대로
        finally:
            tmp.unlink(missing_ok=True)
        if os.name != 'nt':
            # 디렉터리 엔트리 교체까지 디스크에 내려갔음을 보장(best-effort) — Windows/NTFS는
            # 디렉터리 fd를 이런 식으로 열 수 없어 스킵한다(os.replace 자체의 원자성은 유지됨).
            try:
                dir_fd = os.open(str(self.path.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


def inspect_checkpoint(checkpoint_path: str | Path, expected_fingerprint: str) -> CheckpointStatus:
    """compute_zone_motion()을 실제로 돌리지 않고 checkpoint 파일만 열어 유효성을 확인한다.
    worker.py가 "이 job을 재개해도 되는지"를 claim하기 전에 미리 판단할 때 쓴다."""
    _, status = _Checkpoint(checkpoint_path).load(expected_fingerprint)
    return status


def compute_zone_motion(
    video_path: str,
    zones: list[Zone],
    fps: float | None = None,
    frame_stride: int = 1,
    *,
    flow_method: str = 'full_frame',
    checkpoint_path: str | None = None,
    checkpoint_every_frames: int = 300,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """반환: columns = [sec, zone, angle_deg, magnitude_px, motion, low_confidence].

    주의: 각도를 프레임마다 구한 뒤 "각도끼리" 평균 내면 0/360도 경계(예: 358도와 2도)에서
    산술평균이 180도처럼 완전히 틀린 값이 나온다(원형 통계 문제). 그래서 프레임별 각도/크기로
    바꾸기 전의 (mx, my) 벡터 성분 상태로 누적해서 평균 낸 뒤, 마지막에 한 번만 각도/크기로
    변환한다 — 벡터 합산은 경계를 지나도 불연속이 없다.

    frame_stride: 장시간 영상 속도 최적화용. 이 영상의 가장 비싼 연산인
    calcOpticalFlowFarneback()을 매 프레임이 아니라 frame_stride 프레임마다 한 번만 돌리고,
    그 사이 프레임은 직전 계산값을 그대로 쓴다(carry-forward). stride칸 사이 변위를 쓰므로
    벡터를 stride로 나눠 "프레임당 평균 이동"으로 맞춘다 — 그 구간 동안 등속 이동이라는
    가정이 들어가므로, 실제로는 그 안에서 방향이 급히 바뀌면 부정확해진다(shortcut).

    checkpoint_path: 지정하면 checkpoint_every_frames 프레임마다 진행 상태를 저장하고,
    같은 경로에 이미 유효한(같은 영상/구역/설정 서명) checkpoint가 있으면 처음부터 다시
    돌리지 않고 그 지점부터 이어서 계산한다. 정상 종료 시 checkpoint 파일은 삭제한다.
    (적용 범위는 _Checkpoint 클래스 설명 참고 — 같은 프로세스 내 재시도 용도.)

    progress_callback: checkpoint_every_frames마다 (처리한 프레임 수, 전체 프레임 수)로
    호출된다. 전체 프레임 수는 컨테이너 메타데이터 추정치라 완전히 정확하지 않을 수 있다.
    """
    if type(frame_stride) is not int or frame_stride < 1:
        raise ValueError("frame_stride must be a positive int")
    if flow_method not in ('full_frame', 'roi_crop_128'):
        raise ValueError('Invalid flow method')
    columns = ["sec", "zone", "angle_deg", "magnitude_px", "motion", "low_confidence"]
    if not zones:
        return pd.DataFrame(columns=columns)
    cap = cv2.VideoCapture(video_path)
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Only probed when actually needed: callers/tests that mock cv2.VideoCapture narrowly
    # (e.g. only CAP_PROP_FPS/WIDTH/HEIGHT) must keep working when they pass no callback.
    frame_count_hint = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0) if progress_callback else 0
    diag = (w**2 + h**2) ** 0.5  # shortcut: motion 정규화 기준 = 프레임 대각선 길이 (카메라 보정 전 임시 기준)

    masks = {z.name: _zone_mask(z, (h, w)) for z in zones}
    x0, y0, x1, y1 = 0, 0, w, h
    if flow_method == 'roi_crop_128':
        points = np.concatenate([np.asarray(z.polygon) for z in zones])
        x0, y0 = np.maximum(np.floor(points.min(axis=0)).astype(int) - 128, 0)
        x1, y1 = np.minimum(np.ceil(points.max(axis=0)).astype(int) + 129, [w, h])
        x0, y0 = min(int(x0), w), min(int(y0), h)
        x1, y1 = max(int(x1), x0), max(int(y1), y0)
        masks = {name: mask[y0:y1, x0:x1] for name, mask in masks.items()}

    # 매 계산 프레임마다 "mask == 1"을 다시 비교하는 대신, 구역별 인덱스를 한 번만 구해
    # 재사용한다(zone mask 자체는 프레임마다 바뀌지 않으므로 반복 비교가 불필요한 연산이었다).
    zone_indices = {name: np.flatnonzero(mask.reshape(-1)) for name, mask in masks.items()}

    checkpoint = _Checkpoint(checkpoint_path) if checkpoint_path else None
    fingerprint = _checkpoint_fingerprint(video_path, zones, fps, frame_stride, flow_method) if checkpoint else None

    resumed, checkpoint_status = checkpoint.load(fingerprint) if checkpoint else (None, None)
    if checkpoint and not checkpoint_status.valid and checkpoint_status.reason:
        # checkpoint 파일은 있었지만 못 쓴 경우(손상/핑거프린트 불일치 등) — 조용히 무시하지
        # 않고 이유를 남긴다. 같은 프로세스 내 재시도라면 이 call은 그냥 처음부터 다시 돈다.
        print(f'optical flow checkpoint ignored ({checkpoint_status.reason}); starting from frame 0')
    acc: dict[tuple[int, str], list] = {}
    if resumed is not None:
        frame_idx, acc, last_values, prev_gray = resumed
        # frame_idx is the 0-based video frame already processed and folded into acc/prev_gray;
        # the next cap.read() must return the *following* frame, not re-read the same one
        # (re-reading frame_idx would double-count its contribution to acc).
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 1)
    else:
        ok, prev = cap.read()
        if not ok:
            cap.release()
            return pd.DataFrame(columns=columns)
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        frame_idx = 0
        last_values = {z.name: (None, None, True) for z in zones}  # carry-forward for skipped frames

    last_checkpoint_at = frame_idx
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            sec = frame_idx // int(round(fps))

            if frame_idx % frame_stride == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                scene_change = float(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16)).mean()) > SCENE_CHANGE_DIFF
                # Scene-change check intentionally stays in original image coordinates.
                flow = (cv2.calcOpticalFlowFarneback(prev_gray[y0:y1, x0:x1], gray[y0:y1, x0:x1],
                                                    None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        if x1 > x0 and y1 > y0 else None)
                prev_gray = gray
                flat_flow = flow.reshape(-1, 2) if flow is not None else None

                for name, idx in zone_indices.items():
                    if scene_change or flat_flow is None or idx.size < MIN_VECTORS:
                        last_values[name] = (None, None, True)
                        continue
                    vecs = flat_flow[idx]
                    last_values[name] = (float(vecs[:, 0].mean()) / frame_stride,
                                          float(vecs[:, 1].mean()) / frame_stride, False)

            for name in zone_indices:
                mx, my, low = last_values[name]
                key = (sec, name)
                entry = acc.get(key)
                if entry is None:
                    entry = [0.0, 0.0, 0, False]
                    acc[key] = entry
                if not low:
                    entry[0] += mx
                    entry[1] += my
                    entry[2] += 1
                entry[3] = entry[3] or low

            if frame_idx - last_checkpoint_at >= checkpoint_every_frames:
                if checkpoint:
                    checkpoint.save(fingerprint, frame_idx, acc, last_values, prev_gray)
                if progress_callback:
                    progress_callback(frame_idx, frame_count_hint)
                last_checkpoint_at = frame_idx
    finally:
        cap.release()

    # checkpoint.clear()는 루프가 예외 없이 끝났을 때만 실행된다 — 중간에 예외가 나면
    # (예: 디코딩 오류, 메모리 부족) 마지막으로 저장된 checkpoint를 그대로 남겨서,
    # 호출 측이 같은 checkpoint_path로 다시 불러 이어서 계산할 수 있게 한다.
    if checkpoint:
        checkpoint.clear()
    if progress_callback:
        progress_callback(frame_idx, frame_count_hint)

    if not acc:
        return pd.DataFrame(columns=columns)

    sec_vals, zone_vals, mx_vals, my_vals, low_vals = [], [], [], [], []
    for (sec, name), (sum_mx, sum_my, count, any_low) in acc.items():
        sec_vals.append(sec)
        zone_vals.append(name)
        mx_vals.append(sum_mx / count if count else np.nan)
        my_vals.append(sum_my / count if count else np.nan)
        low_vals.append(bool(any_low))

    agg = pd.DataFrame({"sec": sec_vals, "zone": zone_vals, "mx": mx_vals, "my": my_vals,
                        "low_confidence": low_vals})
    magnitude = np.sqrt(agg["mx"] ** 2 + agg["my"] ** 2)
    agg["magnitude_px"] = magnitude
    agg["motion"] = magnitude / diag * 100
    agg["angle_deg"] = np.degrees(np.arctan2(agg["my"], agg["mx"])) % 360
    agg.loc[agg["low_confidence"], ["angle_deg", "magnitude_px", "motion"]] = None
    agg = agg.sort_values(["sec", "zone"], kind="stable").reset_index(drop=True)
    return agg[columns]
