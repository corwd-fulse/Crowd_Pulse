"""실제 영상 2~3분 클립으로 Optical Flow 단계별 비용(디코딩/그레이스케일/Farneback/구역집계)과
frame_stride=1 vs 2 차이를 재는 CLI. 40분 전체를 다시 돌리기 전에, 어디가 진짜 느린지부터
확인하는 용도다 — 정확도에 영향을 주는 설정(stride 등)은 측정만 하고 바꾸지 않는다.

사용 예 (원본에서 2분만 무손실로 잘라서 사용 — 재인코딩 없이 빠르다):
  ffmpeg -ss 00:10:00 -t 120 -i original.mp4 -c copy clip.mp4
  python scripts/profile_optical_flow.py --video clip.mp4 \
      --zone "A:0,0;1920,0;1920,1080;0,1080"

여러 구역을 걸고 싶으면 --zone을 여러 번 준다. --zone을 하나도 안 주면 화면 전체를
단일 구역으로 쓴다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.optical_flow import MIN_VECTORS, SCENE_CHANGE_DIFF, _zone_mask  # noqa: E402
from app.services.zone_analysis import Zone  # noqa: E402


def parse_zone(spec: str) -> Zone:
    name, pts = spec.split(':', 1)
    polygon = [tuple(map(float, p.split(','))) for p in pts.split(';')]
    return Zone(name, polygon)


def print_build_info():
    info = cv2.getBuildInformation()
    print('=== OpenCV build (SIMD/IPP/threading) ===')
    for line in info.splitlines():
        if any(k in line for k in ('SIMD', 'IPP', 'Parallel framework', 'FFMPEG', 'Use Intel')):
            print(' ', line.strip())
    print(f'  cv2.getNumThreads() = {cv2.getNumThreads()}')
    print()


def probe(video_path: str, zones: list[Zone], frame_stride: int, max_frames: int | None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    masks = {z.name: _zone_mask(z, (h, w)) for z in zones}
    zone_indices = {name: np.flatnonzero(mask.reshape(-1)) for name, mask in masks.items()}

    n_to_process = min(total_frames, max_frames) if max_frames else total_frames
    t_decode = t_gray = t_farneback = t_aggregate = 0.0
    farneback_calls = 0
    low_confidence_calls = 0

    ok, prev = cap.read()
    if not ok:
        raise SystemExit('video has no frames')
    t0 = time.perf_counter(); prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY); t_gray += time.perf_counter() - t0

    frame_idx = 0
    wall_start = time.perf_counter()
    while frame_idx < n_to_process - 1:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t_decode += time.perf_counter() - t0
        if not ok:
            break
        frame_idx += 1

        if frame_idx % frame_stride == 0:
            t0 = time.perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t_gray += time.perf_counter() - t0

            t0 = time.perf_counter()
            scene_change = float(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16)).mean()) > SCENE_CHANGE_DIFF
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            t_farneback += time.perf_counter() - t0
            farneback_calls += 1
            prev_gray = gray

            t0 = time.perf_counter()
            flat_flow = flow.reshape(-1, 2)
            for name, idx in zone_indices.items():
                if scene_change or idx.size < MIN_VECTORS:
                    low_confidence_calls += 1
                    continue
                vecs = flat_flow[idx]
                _ = (float(vecs[:, 0].mean()), float(vecs[:, 1].mean()))
            t_aggregate += time.perf_counter() - t0
    cap.release()
    wall = time.perf_counter() - wall_start

    return dict(
        frames_processed=frame_idx, farneback_calls=farneback_calls,
        low_confidence_zone_calls=low_confidence_calls, wall_seconds=wall,
        decode_seconds=t_decode, grayscale_seconds=t_gray, farneback_seconds=t_farneback,
        aggregate_seconds=t_aggregate, fps_width=w, fps_height=h, video_fps=fps,
        total_frames=total_frames, effective_fps=frame_idx / wall if wall else 0.0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video', required=True)
    parser.add_argument('--zone', action='append', default=[],
                        help='"이름:x1,y1;x2,y2;x3,y3[;...]" — 여러 번 지정 가능')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='클립 전체 대신 앞부분 N프레임만 측정(기본: 전체)')
    args = parser.parse_args()

    print_build_info()

    cap = cv2.VideoCapture(args.video)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    zones = [parse_zone(z) for z in args.zone] or [Zone('full_frame', [(0, 0), (w, 0), (w, h), (0, h)])]

    for stride in (1, 2):
        print(f'=== frame_stride={stride} ===')
        result = probe(args.video, zones, stride, args.max_frames)
        for k, v in result.items():
            print(f'  {k}: {v:.3f}' if isinstance(v, float) else f'  {k}: {v}')
        total = result['decode_seconds'] + result['grayscale_seconds'] + result['farneback_seconds'] + result['aggregate_seconds']
        if total:
            print('  --- breakdown of accounted time ---')
            for label, seconds in (('decode', result['decode_seconds']), ('grayscale', result['grayscale_seconds']),
                                   ('farneback', result['farneback_seconds']), ('aggregate', result['aggregate_seconds'])):
                print(f'    {label}: {seconds:.3f}s ({100*seconds/total:.1f}%)')
        extrapolated_min = result['wall_seconds'] / max(result['frames_processed'], 1) * result['total_frames'] / 60
        print(f'  => 이 클립 길이가 전체 영상의 frame 비율로 그대로 늘어난다면, '
              f'전체 영상 예상 소요: 약 {extrapolated_min:.1f}분 (이 클립 기준 선형 추정)')
        print()


if __name__ == '__main__':
    main()
