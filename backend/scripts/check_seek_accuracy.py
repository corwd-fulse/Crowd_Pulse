"""실제 영상에서 cv2.VideoCapture의 CAP_PROP_POS_FRAMES seek이 프레임 정확한지 확인한다.

Optical Flow checkpoint/resume은 중단된 지점부터 이 seek으로 이어서 읽는다. 컨테이너/코덱에
따라 "가장 가까운 키프레임"으로만 맞춰질 수 있다는 우려가 있어서, seek으로 읽은 프레임과
0번부터 순서대로 디코딩해서 같은 번호까지 읽은 프레임을 직접 픽셀 단위로 비교한다.

사용 예(실제 영상으로):
  python scripts/check_seek_accuracy.py --video /path/to/real_clip.mp4

--video를 생략하면 ffmpeg로 합성 H.264(GOP=50, B-frame 포함) 데모 클립을 만들어 보여준다
(실제 영상 없이 메커니즘만 먼저 확인할 때 용도).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def make_demo_clip(path: Path, w=320, h=240, n=400, fps=25, gop=50) -> None:
    raw = path.with_suffix('.raw.mp4')
    rng = np.random.default_rng(9)
    base = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (11, 11), 0)
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for i in range(n):
        frame = np.roll(base, shift=i * 3, axis=1)
        noise = rng.normal(0, 3, (h, w, 3)).astype(np.int16)
        writer.write(np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    writer.release()
    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error', '-i', str(raw),
        '-c:v', 'libx264', '-preset', 'veryfast', '-g', str(gop), '-keyint_min', str(gop),
        '-sc_threshold', '0', '-bf', '2', '-pix_fmt', 'yuv420p', str(path),
    ], check=True)
    raw.unlink()


def read_sequentially_up_to(cap, n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame = None
    for _ in range(n + 1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'sequential decode failed before reaching frame {n}')
    return frame


def read_via_seek(cap, n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f'seek+read failed at frame {n}')
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video', help='실제 영상 경로. 생략하면 합성 H.264 데모 클립 생성')
    parser.add_argument('--probe-every', type=int, default=17,
                        help='전체 프레임 중 이 간격으로도 추가 샘플링(기본 17)')
    args = parser.parse_args()

    if args.video:
        video_path = Path(args.video)
    else:
        video_path = Path(tempfile.gettempdir()) / 'seek_accuracy_demo.mp4'
        print(f'--video 없음: 합성 H.264 데모 클립을 생성합니다 ({video_path})')
        make_demo_clip(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f'cannot open {video_path}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'reported frame count: {total}')
    if total < 2:
        raise SystemExit('video too short to probe')

    # 영상 전체에 걸쳐 고르게 샘플링 + 맨 앞/뒤 몇 프레임은 항상 포함.
    probe = sorted(set(
        list(range(0, min(total, 5))) + list(range(max(0, total - 5), total)) +
        list(range(0, total, max(1, args.probe_every)))
    ))

    mismatches = []
    for idx in probe:
        try:
            seq_frame = read_sequentially_up_to(cap, idx)
            seek_frame = read_via_seek(cap, idx)
        except RuntimeError as exc:
            mismatches.append((idx, str(exc)))
            continue
        if seq_frame.shape != seek_frame.shape:
            mismatches.append((idx, 'shape mismatch'))
            continue
        diff = np.abs(seq_frame.astype(np.int32) - seek_frame.astype(np.int32))
        max_diff = int(diff.max())
        if max_diff != 0:
            mismatches.append((idx, f'max_px_diff={max_diff}, mean={float(diff.mean()):.3f}'))
    cap.release()

    print(f'probed {len(probe)} frame positions; {len(mismatches)} mismatches')
    if mismatches:
        print('MISMATCHES (seek-then-read != sequential decode):')
        for idx, detail in mismatches:
            print(f'  frame {idx}: {detail}')
        print('\n=> CAP_PROP_POS_FRAMES is NOT reliably frame-accurate for this file/backend.')
        print('   Do not trust Optical Flow checkpoint/resume to land exactly on the saved')
        print('   frame for this video; prefer decoding forward from the last keyframe instead.')
        sys.exit(1)
    else:
        print('\n=> All probed frames were pixel-exact. CAP_PROP_POS_FRAMES is frame-accurate')
        print('   for this specific video/container/OpenCV build. Re-run this check against')
        print('   your actual camera/NVR output format before relying on it in production —')
        print('   this only proves accuracy for the file you pointed it at.')


if __name__ == '__main__':
    main()
