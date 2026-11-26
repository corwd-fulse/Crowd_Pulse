"""분석이 끝난 job의 출력 폴더로 시계열 예측(저장소 루트 forecast/ 패키지)을 돌리고 DB에 저장한다.

worker가 --auto-forecast 로 켜져 있으면 job이 succeeded로 저장된 뒤(확장 분석 다음) 자동 실행된다.
실패해도 job 상태는 바꾸지 않는다(확장 분석과 같은 best-effort). 수동 재실행: python -m app.job_forecast.

입력 준비(<out>/forecast/):
  zones.json        roi_snapshot.json의 구역
  zone_meta.json    분 단위 예측용 구역 면적·폭·출구 방향. 실측이 아니라 가정값이다:
                    면적 = 다각형 넓이 / 50 px/m 의 제곱, 폭 = 다각형 가로 폭 / 50, 출구 방향 = 구역 이동 방향의 최빈 10도 구간.
                    그래서 분 단위 밀도·경보는 참고값이다(area_source = assumed).
  yolo/input        pipeline/ 의 crowd_observations·flow·individual_directions·bottleneck_scores
  dmcount/input     인원만 같은 10초 창의 DM-Count 추정치로 바꾼 것(유입·유출은 트래커 값). extended/dmcount_estimates.csv 있을 때
  fused/input       인원 = YOLO + 같은 10초 창의 DM-Count 잔여(YOLO 박스 밖). yolo_residual_count 열이 있을 때
예측 프로세스는 별도 파이썬(forecast 의존성 lightgbm·torch 등이 있는 환경)으로 subprocess 실행한다.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd

from app.services import video_forecast_results as results

ASSUMED_PX_PER_M = 50.0
PIPELINE_FILES = ('crowd_observations', 'flow', 'individual_directions', 'bottleneck_scores')
REPO_ROOT = Path(__file__).resolve().parents[3]  # backend/app/services/job_forecast.py -> 저장소 루트
TIMEOUT_S = 3600


class ForecastSkipped(Exception):
    pass


def zone_meta(zones: list[dict], flow: pd.DataFrame, px_per_m: float = ASSUMED_PX_PER_M) -> dict:
    meta = {}
    for z in zones:
        p = np.array(z['polygon'], float)
        x, y = p[:, 0], p[:, 1]
        area = abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2 / px_per_m ** 2
        width = (x.max() - x.min()) / px_per_m
        a = flow[(flow.zone == z['name']) & flow.angle_deg.notna()]
        if 'low_confidence' in a:
            a = a[~a.low_confidence.astype(str).str.lower().eq('true')]
        hist = np.histogram(a.angle_deg % 360, bins=36, range=(0, 360))[0]
        meta[z['name']] = {'area_m2': float(np.round(area, 1)), 'width_m': float(np.round(width, 1)), 'exit_width_m': float(np.round(width, 1)),
                           'exit_angle_deg': float(hist.argmax() * 10 + 5) if hist.sum() else 0.0, 'shape': 'straight',
                           'polygon': z['polygon'], 'area_source': 'assumed', 'pixels_per_meter': px_per_m}
    return meta


def _window_count(obs: pd.DataFrame, per_window: pd.DataFrame, column: str) -> pd.Series:
    """초별 행에 같은 10초 창(window_start = sec//10*10)의 값을 붙이고, 창이 빠진 초는 앞 창 값으로 채운다."""
    m = obs.assign(window_start=(obs.sec // 10) * 10).merge(per_window[['window_start', 'zone', column]], on=['window_start', 'zone'], how='left')
    m = m.sort_values(['zone', 'sec'])
    m[column] = m.groupby('zone')[column].ffill()
    return m.sort_index()[column]


def prepare(out: Path) -> tuple[Path, list[str]]:
    """job 출력 폴더 -> 예측 작업 폴더(<out>/forecast)와 만들 수 있는 인원 출처 목록."""
    out = Path(out)
    pipe, work = out / 'pipeline', out / 'forecast'
    missing = [n for n in PIPELINE_FILES if not (pipe / f'{n}.csv').is_file()]
    if missing or not (out / 'roi_snapshot.json').is_file():
        raise ForecastSkipped(f'missing pipeline outputs: {missing or ["roi_snapshot.json"]}')
    zones = [{'name': z['name'], 'polygon': z['polygon']} for z in json.loads((out / 'roi_snapshot.json').read_text(encoding='utf-8'))['zones']]
    work.mkdir(exist_ok=True)
    (work / 'zones.json').write_text(json.dumps(zones, ensure_ascii=False, indent=1), encoding='utf-8')
    flow = pd.read_csv(pipe / 'flow.csv')
    (work / 'zone_meta.json').write_text(json.dumps(zone_meta(zones, flow), ensure_ascii=False, indent=1), encoding='utf-8')
    obs = pd.read_csv(pipe / 'crowd_observations.csv')
    dm_csv = out / 'extended' / 'dmcount_estimates.csv'
    dm = pd.read_csv(dm_csv) if dm_csv.is_file() else None
    sources = ['yolo']
    if dm is not None and len(dm):
        sources.append('dmcount')
        if 'yolo_residual_count' in dm and dm.yolo_residual_count.notna().any():
            sources.append('fused')
    for source in sources:
        d = work / source / 'input'
        d.mkdir(parents=True, exist_ok=True)
        for n in PIPELINE_FILES:
            shutil.copyfile(pipe / f'{n}.csv', d / f'{n}.csv')
        if source == 'dmcount':
            c = _window_count(obs, dm, 'estimated_count')
            obs.assign(count=c.fillna(obs['count']).round(2)).to_csv(d / 'crowd_observations.csv', index=False)
        elif source == 'fused':
            r = _window_count(obs, dm, 'yolo_residual_count')
            obs.assign(count=(obs['count'] + r.fillna(0)).round(2)).to_csv(d / 'crowd_observations.csv', index=False)
    return work, sources


def _fps(out: Path, detections: Path) -> float:
    meta = out / 'execution_metadata.json'
    if meta.is_file():
        fps = json.loads(meta.read_text(encoding='utf-8')).get('fps')
        if fps:
            return float(fps)
    d = pd.read_csv(detections, usecols=['frame', 't_ms'])
    d = d[d.t_ms > 0]
    return float((d.frame / (d.t_ms / 1000)).median())


def run_forecast(work: Path, source: str, detections: Path, fps: float, python: str, repo_root: Path,
                 pre_dir: Path | None = None, timeout_s: int = TIMEOUT_S) -> None:
    cmd = [python, '-m', 'forecast.run_job_forecast', str(work), '--source', source, '--detections', str(detections), '--fps', f'{fps:g}']
    if pre_dir is not None:
        cmd += ['--pre-dir', str(pre_dir)]
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONPATH': str(repo_root)}
    with (work / f'run_{source}.log').open('w', encoding='utf-8') as log:
        proc = subprocess.run(cmd, cwd=repo_root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f'forecast {source} exited {proc.returncode} (see {work / f"run_{source}.log"})')


def run_for_job(sessions, job_id: UUID, out: Path, python: str | None = None, repo_root: Path | None = None,
                pre_dir: Path | None = None) -> dict:
    """준비 -> 출처별 예측 -> DB 저장. 출처 하나가 실패해도 나머지는 계속한다. 반환: 출처별 결과/오류."""
    repo = Path(repo_root or REPO_ROOT)
    if not (repo / 'forecast' / 'run_job_forecast.py').is_file():
        raise ForecastSkipped(f'forecast package not found under {repo}')
    out = Path(out)
    detections = out / 'tracker' / 'video_detections.csv'
    if not detections.is_file():
        raise ForecastSkipped('missing tracker/video_detections.csv')
    work, sources = prepare(out)
    fps = _fps(out, detections)
    report = {'job_id': str(job_id), 'fps': fps, 'sources': {}}
    for source in sources:
        t0 = time.monotonic()
        try:
            run_forecast(work, source, detections, fps, python or sys.executable, repo, pre_dir)
            with sessions() as db:
                r = results.import_run(db, job_id, source, work, {'auto': True, 'pre_dir': str(pre_dir) if pre_dir else 'default'})
                db.commit()
            report['sources'][source] = {'points': r['points'], 'level_events': r['level_events'], 'seconds': round(time.monotonic() - t0)}
        except Exception as exc:  # 한 출처 실패가 다른 출처를 막지 않게
            report['sources'][source] = {'error': f'{type(exc).__name__}: {exc}'[:300]}
    return report
