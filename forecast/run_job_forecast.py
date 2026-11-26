"""분석이 끝난 영상 job 하나에 시계열 예측(초 단위 10·30·60초 + 분 단위 5·10·15분 + 이상 흐름 3단계)을 돌린다.

python -m forecast.run_job_forecast <작업 폴더> [--source yolo dmcount] [--fps 30]

작업 폴더 구조(준비 단계에서 만든다):
  zones.json          ROI 구역 목록(roi_snapshot.json 의 zones)
  zone_meta.json      분 단위 예측용 구역 면적·폭·출구 방향(가정값이면 밀도·경보도 가정)
  detections.csv      (선택) 트래커 박스 — 없으면 --detections 로 지정
  <source>/input/     crowd_observations.csv, flow.csv, bottleneck_scores.csv, individual_directions.csv
결과는 <source>/ 에 forecast_video.csv, anomaly_*.csv, forecast_minutes.csv, alarms_minutes.csv, summary.json 으로 남는다.
source 이름은 인원 출처: yolo = YOLO+ByteTrack 구역 인원, dmcount = 같은 10초 창의 DM-Count 추정 인원(유입·유출은 트래커 값).
fused = YOLO 인원 + 같은 10초 창의 DM-Count 잔여(YOLO 박스 밖 density, 대시보드 'YOLO + DM-Count(보완)'와 같은 값).
--source fused 는 <work>/fused/input 을 직접 준비해야 해서 기본 실행 목록에는 넣지 않는다.
"""
import argparse, json
from pathlib import Path

import pandas as pd

from .serving.video_pipeline import run_video_forecast


def run(work, source, detections, fps, horizons=(10, 30, 60), pre_dir=None):
    work = Path(work)
    out = work / source
    zones = json.loads((work / "zones.json").read_text(encoding="utf-8"))
    res = run_video_forecast(out / "input", out_dir=out, horizons=horizons, detections_csv=detections, zones=zones, fps=fps,
                             zone_meta=work / "zone_meta.json", job_id=work.parent.name, pre_dir=pre_dir)
    info = res["info"]
    (out / "summary.json").write_text(json.dumps(_jsonable(info), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return res


def _jsonable(x):
    """info 안의 튜플 키(예: status_counts의 (horizon, model))를 'a/b' 문자열로."""
    if isinstance(x, dict):
        return {"/".join(map(str, k)) if isinstance(k, tuple) else k: _jsonable(v) for k, v in x.items()}
    return [_jsonable(v) for v in x] if isinstance(x, list) else x


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("work")
    ap.add_argument("--source", nargs="+", default=["yolo", "dmcount"])
    ap.add_argument("--detections", required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--pre-dir", help="미리 학습 모델 폴더(기본 serving/artifacts/video). 이 영상을 뺀 영상들로 학습한 폴더를 주면 "
                    "lightgbm_pre·lstm_pre 점수가 처음 보는 영상 기준이 된다(학습에 쓴 영상을 다시 채점하면 부풀려짐)")
    a = ap.parse_args()
    for s in a.source:
        res = run(a.work, s, a.detections, a.fps, pre_dir=a.pre_dir)
        i = res["info"]
        print(f"[{s}] 영상 {i['seconds']}초, 구역 {i['zones']} | 초 단위 {i['rows']}행 | 이상 이벤트 {i['anomaly_events']}건 "
              f"(단계 {i['level_events']}) | IF: {i.get('iforest')} | 병목: {i.get('bottleneck')}")
        print(pd.DataFrame(i["backtest"]).T.to_string())
        m = i.get("minutes")
        if m:
            print(f"[{s}] 분 단위 {m['rows']}행, 경보 {m['alarms']}건, 모델 {m['model_version']}")
            print(pd.DataFrame(m["backtest"]).T.to_string())


if __name__ == "__main__":
    main()
