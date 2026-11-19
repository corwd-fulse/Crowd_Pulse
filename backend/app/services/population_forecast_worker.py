"""Private JSON-lines inference process; original forecast modules remain untouched."""
import base64
import contextlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# A direct script launch adds services/ to sys.path. Its bottleneck.py shadows
# pandas' optional third-party bottleneck package; keep only package roots here.
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'backend'))


def infer(request):
    from app.services.population_forecast import artifact_catalog
    meta = artifact_catalog()
    if meta['manifest_sha256'] != request['manifest_sha256']:
        raise ValueError('모델 manifest가 변경됐습니다. 다시 요청하세요.')
    # Importing serving also imports video_pipeline, but never calls/trains it.
    from forecast.serving.pipeline import run_forecast, load_history, get_predictor
    import torch
    torch.set_num_threads(1)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='crowd_population_') as folder:
        source = Path(folder)/('history'+request['suffix'])
        source.write_bytes(base64.b64decode(request['content']))
        wide, _ = load_history(source)
        if len(wide) > 24*366 or wide.shape[1] > 427:
            raise ValueError('입력은 최대 366일·427개 동으로 제한됩니다.')
        # Explicit new temporary output: never write beside a user's input CSV.
        result = run_forecast(source, out_csv=Path(folder)/'predictions.csv', models=('lightgbm', 'lstm'), device='cpu')
        current = wide.iloc[-1]
        def number(value):
            return float(value) if value is not None and math.isfinite(float(value)) else None
        items = []
        for row in result['predictions'].itertuples(index=False):
            if row.area_id not in wide.columns:
                continue  # Predictor returns all 427; do not invent unsubmitted areas.
            observed, value = number(current.get(row.area_id)), number(row.value)
            status = row.status
            if status != 'ok':
                value = None
            elif value is None or value < 0:
                value, status = None, 'invalid_prediction'
            change = None if observed is None or value is None else round(value-observed, 1)
            items.append({'area_id': row.area_id, 'model': row.model, 'model_version': row.model_version,
                          'last_observed_at': row.last_observed_at.isoformat(), 'target_at': row.target_at.isoformat(),
                          'current_value': observed, 'value': value, 'target_actual': number(row.actual), 'status': status,
                          'change': change, 'change_percent': None if change is None or observed == 0 else round(change/observed*100, 2)})
        return {'kind': 'administrative_population_estimate_forecast', 'timezone': 'Asia/Seoul',
                'horizon_hours': 1, 'manifest_sha256': meta['manifest_sha256'], 'items': items,
                'info': {**result['info'], 'input_is_live': False, 'source': 'user_supplied_history',
                         'inference_seconds': time.perf_counter()-started, 'predictor_cache': get_predictor.cache_info()._asdict(),
                         'comparison': '마지막 입력 생활인구 추정치와 1시간 후 예측 비교. 현재 현장 실측이나 영상 ROI 인원이 아닙니다.'}}


def main():
    for line in sys.stdin:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = infer(json.loads(line))
        except (ValueError, FileNotFoundError, KeyError, IndexError) as exc:
            result = {'error': {'code': 'invalid_history', 'message': str(exc)[:300], 'http_status': 422}}
        except ImportError as exc:
            result = {'error': {'code': 'dependency_unavailable', 'message': f'예측 전용 프로세스의 의존성을 확인하세요: {exc.name or type(exc).__name__}'}}
        except Exception:
            result = {'error': {'code': 'inference_failed', 'message': '모델 추론에 실패했습니다. 값을 추정하지 않습니다.'}}
        print(json.dumps(result, default=str, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
