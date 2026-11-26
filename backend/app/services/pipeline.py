"""영상 하나를 분석해서 결과 CSV들을 만드는 단일 진입점.

FastAPI 라우터나 CLI에서 이 함수 하나만 호출하면 구역 통계·Optical Flow·개인별
이동 방향·병목 점수를 전부 계산해 out_dir에 CSV로 저장한다.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import pandas as pd

from .bottleneck import compute_bottleneck_scores
from .bottleneck_diagnostics import describe_bottleneck_windows
from .crowd_speed import PixelsPerMeter, compute_individual_speed, compute_zone_speed
from .density_heatmap import compute_density_heatmap
from .hybrid_flow import build_hybrid_summary
from .individual_flow import DEFAULT_WINDOW_FRAMES, aggregate_by_zone, compute_counter_flow, compute_individual_directions
from .optical_flow import compute_zone_motion
from .queue_analysis import compute_queue_scores
from .tracking_eval import evaluate_tracking
from .zone_analysis import Zone, compute_dwell_times, compute_zone_timeseries


def _run_zone_pipeline(
    video_path: str,
    detections_csv: str,
    zones: list[Zone],
    out_dir: str,
    *, analysis_profile: str = 'full', frame_stride: int = 1,
    flow_method: str = 'full_frame',
    checkpoint_path: str | None = None,
    progress_callback=None,
    flow_workers: int = 1,
    _stack=None,
) -> dict[str, str]:
    """반환: {결과종류: 저장된 csv 경로}.

    video_path: optical_flow 계산에 쓰는 원본 영상(detections.csv 좌표만으론 안 됨).
    detections_csv: tracker.py가 만든 frame,t_ms,track_id,x1,y1,x2,y2,conf 형식 csv.
    frame_stride: 장시간 FULL 분석에서 Optical Flow 비용을 줄이는 옵션(optical_flow.py 참고).
    기본값 1은 예전과 100% 동일하게 매 프레임 계산한다. flow.csv의 컬럼 모양은 그대로다.

    checkpoint_path: 지정하면 Optical Flow 단계(가장 비싼 연산, 중간에 예외로 끊길 가능성이
    제일 큰 구간)가 예외로 끊겼을 때 처음부터 다시 돌리지 않고, 같은 호출 안에서 딱 한 번
    checkpoint에서 재개를 시도한다. worker.py의 job 재시도 정책과는 무관하다 — 같은
    프로세스·같은 호출 안에서의 재시도일 뿐이다(optical_flow.py의 _Checkpoint 설명 참고).
    progress_callback: Optical Flow 진행 상황을 (처리한 프레임 수, 전체 프레임 수)로 받는다.
    flow_workers: Optical Flow를 나눠 계산할 프로세스 수(optical_flow.compute_zone_motion의
    workers). 기본값 1은 예전과 동일한 단일 코어 계산이다.
    """
    if analysis_profile not in ('fast', 'full'):
        raise ValueError('Invalid analysis profile')
    if flow_method not in ('full_frame', 'roi_crop_128'):
        raise ValueError('Invalid flow method')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width,height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    started = time.perf_counter()
    streamed = Path(detections_csv).is_file() and Path(detections_csv).stat().st_size > 8*1024**2
    if streamed:
        from .streaming_tracks import process
        stack=_stack
        zone_ts,individual_zone_agg,prepared_votes=stack.enter_context(process(detections_csv,zones,fps,out / "individual_directions.csv"))
    else:
        zone_ts = compute_zone_timeseries(detections_csv, zones, fps=fps)
    flow_started = time.perf_counter()
    if analysis_profile == 'full':
        flow_kwargs = dict(fps=fps, frame_stride=frame_stride, flow_method=flow_method,
                           checkpoint_path=checkpoint_path, progress_callback=progress_callback,
                           workers=flow_workers)
        try:
            motion_ts = compute_zone_motion(video_path, zones, **flow_kwargs)
        except Exception as exc:
            if checkpoint_path is None:
                raise
            # 가장 비싼 단계에서 끊긴 경우, 처음부터 다시 돌리지 않고 같은 호출 안에서
            # 딱 한 번 checkpoint 지점부터 재개를 시도한다. 재개도 실패하면 그대로 올려서
            # job 실패 처리는 기존 정책(worker.py)을 그대로 따른다 — job 자체를 재시도하지 않는다.
            print(f'optical flow interrupted ({exc!r}); resuming once from checkpoint {checkpoint_path}')
            motion_ts = compute_zone_motion(video_path, zones, **flow_kwargs)
    else:
        motion_ts = pd.DataFrame(columns=['sec','zone','angle_deg','magnitude_px','motion','low_confidence'])
    flow_seconds = time.perf_counter() - flow_started
    if not streamed:
        individual = compute_individual_directions(detections_csv)
        individual_zone_agg = aggregate_by_zone(individual, detections_csv, zones, fps=fps)

    if analysis_profile == 'full':
        merged = zone_ts.merge(motion_ts[["sec", "zone", "motion"]], on=["sec", "zone"], how="left")
        scores = compute_bottleneck_scores(merged, missing_policy='omit_unavailable_v1')
    else:
        scores = pd.DataFrame(columns=['window_start','zone','score','growth','slowdown','imbalance','risk_level'])

    paths = {
        "crowd_observations": out / "crowd_observations.csv",
        "flow": out / "flow.csv",
        "individual_directions": out / "individual_directions.csv",
        "individual_zone_agg": out / "individual_zone_agg.csv",
        "bottleneck_scores": out / "bottleneck_scores.csv",
    }
    zone_ts.to_csv(paths["crowd_observations"], index=False)
    motion_ts.to_csv(paths["flow"], index=False)
    if not streamed:individual.to_csv(paths["individual_directions"], index=False)
    individual_zone_agg.to_csv(paths["individual_zone_agg"], index=False)
    scores.to_csv(paths["bottleneck_scores"], index=False)

    try:
        hybrid = build_hybrid_summary(None if streamed else individual, detections_csv, zones, motion_ts, zone_ts,
                                      **({"prepared_votes":prepared_votes} if streamed else {}))
    finally:
        if streamed:stack.close()
    score_keys = {(int(r.window_start), r.zone) for r in scores.itertuples()}
    diagnostic_rows = zone_ts.merge(motion_ts[['sec', 'zone', 'motion', 'low_confidence']],
                                    on=['sec', 'zone'], how='left').to_dict(orient='records')
    hybrid_rows = {(r['sec'], r['zone']): r for r in hybrid.get('rows', [])}
    for row in diagnostic_rows:
        row['hybrid'] = hybrid_rows.get((row['sec'], row['zone']))
    hybrid['bottleneck_status'] = describe_bottleneck_windows(
        diagnostic_rows, score_keys, analysis_profile=analysis_profile)
    metadata = dict(flow_method=flow_method if analysis_profile == 'full' else 'not_run',
                    requested_flow_method=flow_method, crop_padding_px=128 if flow_method == 'roi_crop_128' else None,
                    bottleneck_formula='legacy-weights-v1', bottleneck_missing_policy='omit_unavailable_v1',
                    individual_speed_px_semantics='displacement over observation window; not px/frame or m/s',
                    farneback_parameters=dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                             poly_n=5, poly_sigma=1.2, flags=0),
                    flow_workers=flow_workers if analysis_profile == 'full' else None,
                    flow_seconds=flow_seconds, zone_pipeline_seconds=time.perf_counter() - started,
                    hybrid=hybrid)
    if Path(detections_csv).is_file():
        from .transition_artifact import summarize_tracking
        metadata['transitions'] = summarize_tracking(detections_csv,zones,fps,frame_count,width,height)
        (out/'zone_transitions.json').write_text(json.dumps(metadata['transitions'],allow_nan=False),encoding='utf-8')
    # Sidecar is copied into execution_snapshot.settings by the Worker. The five
    # existing CSV headers and result manifest remain exactly schema v1.
    (out / 'analysis_details.json').write_text(json.dumps(metadata, allow_nan=False), encoding='utf-8')

    return {k: str(v) for k, v in paths.items()}


def run_zone_pipeline(video_path, detections_csv, zones, out_dir, *, analysis_profile="full",
                      frame_stride=1, flow_method="full_frame", checkpoint_path=None,
                      progress_callback=None, flow_workers=1):
    from contextlib import ExitStack
    with ExitStack() as stack:
        return _run_zone_pipeline(video_path, detections_csv, zones, out_dir,
                                  analysis_profile=analysis_profile, frame_stride=frame_stride,
                                  flow_method=flow_method, checkpoint_path=checkpoint_path,
                                  progress_callback=progress_callback, flow_workers=flow_workers,
                                  _stack=stack)


def run_extended_zone_pipeline(
    video_path: str,
    detections_csv: str,
    zones: list[Zone],
    out_dir: str,
    *,
    pixels_per_meter: PixelsPerMeter | None = None,
    zone_timeseries: pd.DataFrame | None = None,
    motion_timeseries: pd.DataFrame | None = None,
    individual_directions: pd.DataFrame | None = None,
    frame_stride: int = 1,
    dmcount_model_path: str | None = None,
    dmcount_device: str = 'cpu',
    dmcount_gamma: float = 1.0,
    dmcount_input_scale: float = 1.0,
    resumable: bool = False,
    dmcount_half: bool = False,
) -> dict[str, str]:
    """밀집 히트맵·체류 시간·역방향 비율·대기열 점수·추적 품질 평가, 별도 CSV 산출.

    pixels_per_meter: 카메라 보정값이 있을 때만 전달(crowd_speed.py). 없으면(기본값) 속도는
    m/s로 환산하지 않고 그대로 생략한다 — 보정 없이 환산하면 틀린 값을 사실처럼 내보내게 된다.

    dmcount_model_path: DM-Count(density_estimation.py) pretrained 체크포인트 경로가 있을
    때만 전달(기본값 None=실행 안 함). YOLO와 완전히 독립된 별도 모델이고 CPU 추론이 느려서
    (윈도우당 수 초~수십 초) 대부분의 job에서는 켜지 않는 opt-in 기능이다. 정확도가
    YOLO 대비 검증되지 않았으므로(실측 실험에서 영상 조건에 따라 10~120% 가까이 벌어짐)
    bottleneck_scores 등 어떤 계산에도 입력으로 쓰지 않고, 대시보드 참고 표시 전용이다.

    zone_timeseries/motion_timeseries/individual_directions: run_zone_pipeline()이 이미 계산한
    같은 값이 있으면 전달해서 재계산을 피한다. 특히 motion_timeseries(Optical Flow, Farneback)는
    이 파이프라인에서 제일 비싼 연산이라 — FULL 분석에서 run_zone_pipeline()이 한 번 돌린 뒤
    여기서 또 돌리면 그 비싼 연산을 영상 전체에 대해 두 번 하게 된다. 장시간 영상 속도 최적화
    (frame_stride)를 넣어놓고 여기서 또 전체 재계산하면 그 의미가 없어지므로, worker.py는 반드시
    core 파이프라인이 만든 CSV를 읽어 넘겨준다. 직접 호출(CLI/테스트)에서 생략하면 그냥 다시
    계산한다 — 결과는 같고 느릴 뿐이다.

    run_zone_pipeline()의 다섯 결과 파일(crowd_observations/flow/individual_directions/
    individual_zone_agg/bottleneck_scores)은 worker.py가 analysis_jobs DB 저장 시 정확히
    이 다섯 개 키만 있는지 검증한다(app/worker.py의 set(paths)!=set(SCHEMAS) 체크).
    그 계약을 깨지 않도록 이 함수는 완전히 별도의 파일/딕셔너리로 결과를 낸다. worker.py가
    이 결과를 vision_extended_results.py를 통해 별도 테이블(vision_extended_results)에 저장하며,
    그 저장은 core 결과 저장이 끝난 뒤에만 일어나고 실패해도 job 상태에는 영향을 주지 않는다.

    resumable: True면 out_dir에 단계별 진행 기록(EXTENDED_PROGRESS_NAME)을 남겨, 중간에
    끊긴 뒤 같은 out_dir로 다시 호출하면 끝난 단계는 건너뛴다. (1) DM-Count를 뺀 나머지
    지표는 한 묶음으로 끝나는 즉시 파일 sha256과 함께 기록되고, (2) DM-Count는 10초 윈도우마다
    체크포인트(DMCOUNT_CHECKPOINT_NAME)를 남긴다. (1)이 이미 끝난 out_dir이면 detections_csv가
    없어도 된다 — 그래서 out_dir만 다른 컴퓨터로 복사해 DM-Count만 다른 GPU로 이어 계산할 수
    있다. 기본값 False는 기존 동작과 동일(진행 기록 파일을 만들지 않음).

    dmcount_half: DM-Count를 fp16으로 추론(작은 GPU용, CUDA 전용 — density_estimation 참고).
    어느 정밀도로 몇 개 윈도우를 계산했는지는 진행 기록의 meta에 남긴다.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    progress = _ExtendedProgress(out, _extended_base_fingerprint(
        video_path, zones, fps=fps, width=width, height=height,
        pixels_per_meter=pixels_per_meter, frame_stride=frame_stride,
    )) if resumable else None
    paths = progress.completed('base') if progress is not None else None
    if paths is not None:
        print(f'extended: base metrics already completed in {out}; skipping them', flush=True)
    else:
        paths = _compute_extended_base(
            video_path, detections_csv, zones, out, fps=fps, width=width, height=height,
            pixels_per_meter=pixels_per_meter, zone_timeseries=zone_timeseries,
            motion_timeseries=motion_timeseries, individual_directions=individual_directions,
            frame_stride=frame_stride,
        )
        if progress is not None:
            progress.mark('base', paths)

    if dmcount_model_path is not None:
        from .density_estimation import compute_dmcount_estimates
        # 보정 설정을 model_version에 남겨 대시보드/DB에서 어떤 입력 조건의 추정치인지 구분한다.
        model_version = "dmcount_qnrf"
        if dmcount_gamma != 1.0 or dmcount_input_scale != 1.0:
            model_version += f"_g{dmcount_gamma:g}_s{dmcount_input_scale:g}"
        # 추적 결과(박스)가 있으면 YOLO 박스 밖 잔여 density도 같이 구한다(YOLO+DM-Count 보완 추정).
        # 이 단계 이름에 표시해서, 잔여 없이 끝난 예전 DM-Count 단계를 완료로 착각하지 않게 한다.
        # frame_stride>1이면 DM-Count 샘플 프레임에 박스가 없을 수 있어(박스 0개 = 전부 잔여로 오인) 끈다.
        fuse_detections = (Path(detections_csv) if frame_stride == 1 and detections_csv
                           and Path(detections_csv).is_file() else None)
        stage = f'dmcount:{model_version}' + (':residual' if fuse_detections is not None else '')
        done = progress.completed(stage) if progress is not None else None
        if done is not None:
            print('extended: DM-Count already completed; skipping it', flush=True)
            paths.update(done)
        else:
            dmcount = compute_dmcount_estimates(video_path, zones, fps=fps, model_path=dmcount_model_path, device=dmcount_device,
                                                 model_version=model_version, gamma=dmcount_gamma,
                                                 input_scale=dmcount_input_scale,
                                                 heatmap_dir=out / "dmcount_heatmaps",
                                                 checkpoint_path=out / DMCOUNT_CHECKPOINT_NAME if resumable else None,
                                                 half=dmcount_half,
                                                 detections_csv=fuse_detections)
            dmcount_paths = {"dmcount_estimates": out / "dmcount_estimates.csv"}
            dmcount.to_csv(dmcount_paths["dmcount_estimates"], index=False)
            precisions = dmcount.attrs.get("windows_by_precision", {})
            print(f'extended: DM-Count windows by precision {precisions}', flush=True)
            if progress is not None:
                progress.mark(stage, dmcount_paths, meta={'windows_by_precision': precisions})
            paths.update(dmcount_paths)
    return {k: str(v) for k, v in paths.items()}


EXTENDED_PROGRESS_NAME = 'extended_progress.json'
DMCOUNT_CHECKPOINT_NAME = 'dmcount_checkpoint.json'


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _extended_base_fingerprint(video_path: str, zones: list[Zone], **settings) -> str:
    # 영상 경로/mtime은 넣지 않는다 — out_dir을 다른 컴퓨터로 옮겨 이어가는 경우를 허용하기
    # 위해서다(density_estimation._dmcount_fingerprint와 같은 이유). 같은 영상인지는 크기로 본다.
    payload = dict(video_size=os.path.getsize(video_path),
                   zones=[(z.name, z.polygon) for z in zones], **settings)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class _ExtendedProgress:
    """run_extended_zone_pipeline(resumable=True)의 단계별 완료 기록.

    stage마다 {kind: (out_dir 기준 상대 파일명, sha256)}를 저장하고, 다시 읽을 때 서명이 같고
    파일들이 그대로(같은 sha256) 있을 때만 "완료"로 인정한다. 하나라도 어긋나면 그 단계는
    처음부터 다시 계산한다(조용히 깨진 결과를 재사용하지 않도록).
    """

    def __init__(self, out: Path, fingerprint: str):
        self.out = out
        self.path = out / EXTENDED_PROGRESS_NAME
        state = None
        if self.path.is_file():
            try:
                state = json.loads(self.path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                state = None
        if not state or state.get('fingerprint') != fingerprint:
            if state:
                print('extended: progress record does not match this video/zones/settings; recomputing', flush=True)
            state = {'fingerprint': fingerprint, 'stages': {}}
        self.state = state

    def completed(self, stage: str) -> dict[str, Path] | None:
        files = self.state['stages'].get(stage)
        if files is None:
            return None
        paths = {}
        for kind, info in files.items():
            path = self.out / info['file']
            if not path.is_file() or _sha256_file(path) != info['sha256']:
                print(f'extended: {stage} output {info["file"]} missing or changed; recomputing {stage}', flush=True)
                return None
            paths[kind] = path
        return paths

    def mark(self, stage: str, paths: dict[str, Path], meta: dict | None = None) -> None:
        self.state['stages'][stage] = {
            kind: {'file': Path(path).resolve().relative_to(self.out.resolve()).as_posix(),
                   'sha256': _sha256_file(Path(path))}
            for kind, path in paths.items()
        }
        if meta is not None:
            self.state.setdefault('meta', {})[stage] = meta
        tmp = self.path.with_name(self.path.name + '.tmp')
        tmp.write_text(json.dumps(self.state, indent=2), encoding='utf-8')
        os.replace(tmp, self.path)


def _compute_extended_base(
    video_path: str,
    detections_csv: str,
    zones: list[Zone],
    out: Path,
    *,
    fps: float,
    width: int,
    height: int,
    pixels_per_meter: PixelsPerMeter | None,
    zone_timeseries: pd.DataFrame | None,
    motion_timeseries: pd.DataFrame | None,
    individual_directions: pd.DataFrame | None,
    frame_stride: int,
) -> dict[str, Path]:
    """DM-Count를 뺀 확장 지표 묶음(체류/역방향/대기열/밀집 히트맵/추적 품질/속도)."""
    # detections.csv를 한 번만 읽어 아래 네 함수 모두에 DataFrame으로 넘긴다 — 안 그러면
    # dwell/counter_flow/density_heatmap/tracking_quality가 각자 같은 파일을 다시 파싱한다
    # (장시간 영상일수록 그 비용이 커지는, Optical Flow 이중계산과 같은 종류의 비효율).
    det = pd.read_csv(detections_csv)

    dwell = compute_dwell_times(det, zones, fps=fps)
    individual = individual_directions if individual_directions is not None else compute_individual_directions(det)
    counter_flow = compute_counter_flow(individual, det, zones, fps=fps)

    zone_ts = zone_timeseries if zone_timeseries is not None else compute_zone_timeseries(det, zones, fps=fps)
    motion_ts = motion_timeseries if motion_timeseries is not None else compute_zone_motion(video_path, zones, fps=fps, frame_stride=frame_stride)
    merged = zone_ts.merge(motion_ts[["sec", "zone", "motion"]], on=["sec", "zone"], how="left")
    queue_scores = compute_queue_scores(merged)

    heatmap = compute_density_heatmap(det, width, height, fps=fps)
    tracking_quality = evaluate_tracking(det)

    paths = {
        "dwell_times": out / "dwell_times.csv",
        "counter_flow": out / "counter_flow.csv",
        "queue_scores": out / "queue_scores.csv",
        "density_heatmap": out / "density_heatmap.csv",
        "tracking_quality": out / "tracking_quality.json",
    }
    dwell.to_csv(paths["dwell_times"], index=False)
    counter_flow.to_csv(paths["counter_flow"], index=False)
    queue_scores.to_csv(paths["queue_scores"], index=False)
    heatmap.to_csv(paths["density_heatmap"], index=False)
    paths["tracking_quality"].write_text(json.dumps(tracking_quality, indent=2, ensure_ascii=False))

    if pixels_per_meter is not None:
        zone_speed = compute_zone_speed(motion_ts, fps=fps, pixels_per_meter=pixels_per_meter)
        paths["crowd_speed_zone"] = out / "crowd_speed_zone.csv"
        zone_speed.to_csv(paths["crowd_speed_zone"], index=False)

        # 개인별 속도는 구역 하나짜리 단일 비율(float)일 때만 낸다. dict(구역별 비율)로 받았을 때
        # 어느 비율을 적용할지는 그 순간 그 사람이 있는 구역을 다시 조회해야 해서, 여기서는
        # 섣불리 아무 값이나 고르지 않고 생략한다.
        if isinstance(pixels_per_meter, (int, float)):
            individual_speed = compute_individual_speed(
                individual, fps=fps, pixels_per_meter=pixels_per_meter, window_frames=DEFAULT_WINDOW_FRAMES
            )
            paths["crowd_speed_individual"] = out / "crowd_speed_individual.csv"
            individual_speed.to_csv(paths["crowd_speed_individual"], index=False)
    return paths
