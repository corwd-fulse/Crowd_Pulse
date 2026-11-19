"""Isolated CPU-only adapter. Never imports ML libraries in the API process."""
import atexit
import base64
import hashlib
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / 'forecast/serving/artifacts'
DEPENDENCIES = ('numpy', 'pandas', 'torch', 'lightgbm', 'holidays', 'sklearn')
CREATE_FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def configured_python():
    """Environment override, then optional local configuration; never activate an env."""
    if os.environ.get('CROWD_FORECAST_PYTHON'):
        return os.environ['CROWD_FORECAST_PYTHON']
    config = ROOT / 'backend/forecast.runtime.local'
    if config.exists():
        try:
            path = Path(json.loads(config.read_text(encoding='utf-8'))['python'])
            return str((config.parent / path).resolve() if not path.is_absolute() else path)
        except (OSError, ValueError, KeyError, TypeError):
            return str(config.parent / '__invalid_forecast_runtime__')
    return sys.executable


class ForecastError(Exception):
    def __init__(self, code, message, http_status=503):
        self.code, self.message, self.http_status = code, message, http_status
        super().__init__(message)


def artifact_catalog():
    try:
        manifest_bytes = (ARTIFACTS / 'manifest.json').read_bytes()
        manifest = json.loads(manifest_bytes)
        entries = [manifest[k] for k in ('scale', 'dong_order', 'feature_cols')]
        entries += [v for v in manifest['models'].values() if v.get('file')]
        for entry in entries:
            path = (ARTIFACTS / entry['file']).resolve()
            if path.parent != ARTIFACTS.resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('Artifact hash mismatch')
        areas = json.loads((ARTIFACTS / manifest['dong_order']['file']).read_text())
        if len(areas) != manifest['areas'] or len(set(areas)) != len(areas):
            raise ValueError('Area order mismatch')
        return {'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                'areas': areas, 'lookback_hours': manifest['lookback_h'], 'horizon_hours': 1,
                'models': {k: v['version'] for k, v in manifest['models'].items() if k in ('lightgbm', 'lstm')},
                'train_period': manifest['train_period'], 'test_period': manifest['test_period']}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ForecastError('artifact_unavailable', '모델 파일 또는 manifest 무결성을 확인할 수 없습니다.') from exc


def runtime_status():
    executable = configured_python()
    if Path(executable).resolve() == Path(sys.executable).resolve():
        missing = [n for n in DEPENDENCIES if importlib.util.find_spec(n) is None]
    else:
        try:
            code = 'import importlib.util,json;print(json.dumps([n for n in '+repr(DEPENDENCIES)+' if importlib.util.find_spec(n) is None]))'
            result = subprocess.run([executable, '-B', '-c', code], capture_output=True, text=True, timeout=10, creationflags=CREATE_FLAGS)
            missing = json.loads(result.stdout) if result.returncode == 0 else list(DEPENDENCIES)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            missing = list(DEPENDENCIES)
    return executable, missing


def catalog():
    try:
        result = artifact_catalog()
    except ForecastError as exc:
        return {'status': 'unavailable', 'code': exc.code, 'message': exc.message, 'areas': [], 'models': {}}
    _, missing = runtime_status()
    result.update(status='unavailable' if missing else 'ready',
                  code='dependency_unavailable' if missing else 'ready', missing_dependencies=missing,
                  message='예측 전용 Python 의존성이 부족합니다. 환경을 자동 변경하지 않습니다.' if missing else '생활인구 1시간 후 예측 준비 완료',
                  input_formats=['population_history CSV', '서울 생활인구 CSV', 'ZIP'],
                  area_names_available=False, video_forecast_enabled=False)
    state = broker.model_state
    if missing or broker.signature not in (None, (configured_python(), result['manifest_sha256'])):
        state = 'not_loaded'
    elif broker.process is not None and broker.process.poll() is not None and state == 'loaded':
        state = 'failed'
    result.update(model_state=state, last_error_code=broker.last_error_code,
                  availability_basis='artifact_hash_and_module_discovery; model_state_tracks_actual_worker_load')
    return result


class ForecastBroker:
    """One persistent subprocess, bounded single-flight cache, no video Worker use."""
    def __init__(self):
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='population-forecast')
        self.pending = {}
        self.cache = OrderedDict()
        self.process = None
        self.signature = None
        self.output = None
        self.closed = False
        self.model_state = 'not_loaded'
        self.last_error_code = None

    def submit(self, content: bytes, suffix: str):
        meta = artifact_catalog()
        executable, missing = runtime_status()
        if missing:
            raise ForecastError('dependency_unavailable', '필수 예측 의존성 부족: '+', '.join(missing))
        signature = (executable, meta['manifest_sha256'])
        key = hashlib.sha256(content+suffix.encode()+repr(signature).encode()).hexdigest()
        with self.lock:
            if self.closed:
                raise ForecastError('service_closed', '예측 서비스가 종료 중입니다.')
            if key in self.cache:
                from concurrent.futures import Future
                self.cache.move_to_end(key)
                f = Future(); f.set_result({**self.cache[key], 'cache_hit': True})
                return f
            if key in self.pending:
                return self.pending[key]
            if len(self.pending) >= 4:
                raise ForecastError('busy', '예측 요청이 대기 중입니다. 잠시 후 다시 시도하세요.', 429)
            f = self.executor.submit(self._run, content, suffix, signature)
            self.pending[key] = f
        def done(future):
            with self.lock:
                self.pending.pop(key, None)
                if not future.cancelled() and future.exception() is None:
                    self.cache[key] = future.result()
                    while len(self.cache) > 8:
                        self.cache.popitem(last=False)
        f.add_done_callback(done)
        return f

    def _stop(self):
        p = self.process
        self.process = None
        if p is not None:
            if p.poll() is None:
                p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(timeout=5)
            for pipe in (p.stdin, p.stdout):
                if pipe:
                    pipe.close()

    def _start(self, signature):
        self._stop()
        self.model_state = 'loading'
        env = os.environ.copy()
        # Child-only limits: do not change the API/Worker's runtime or environments.
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        self.process = subprocess.Popen([signature[0], '-B', str(Path(__file__).with_name('population_forecast_worker.py'))],
            cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding='utf8', creationflags=CREATE_FLAGS)
        self.signature = signature
        output = self.output = queue.Queue()
        def reader(pipe):
            try:
                for line in pipe:
                    output.put(line)
            finally:
                output.put(None)
        threading.Thread(target=reader, args=(self.process.stdout,), daemon=True).start()

    def _run(self, content, suffix, signature):
        try:
            if self.process is None or self.process.poll() is not None or self.signature != signature:
                self._start(signature)
            self.process.stdin.write(json.dumps({'content': base64.b64encode(content).decode(), 'suffix': suffix, 'manifest_sha256': signature[1]})+'\n')
            self.process.stdin.flush()
            line = self.output.get(timeout=90)
            if line is None:
                raise ForecastError('process_failed', '예측 전용 프로세스가 종료됐습니다.')
            result = json.loads(line)
            if not isinstance(result, dict):
                raise ValueError('Invalid worker response')
            if 'error' in result:
                raise ForecastError(result['error']['code'], result['error']['message'], result['error'].get('http_status', 503))
            self.model_state = 'loaded'
            self.last_error_code = None
            return {**result, 'cache_hit': False}
        except queue.Empty as exc:
            self._stop()
            self.model_state = 'failed'; self.last_error_code = 'timeout'
            raise ForecastError('timeout', '예측 제한 시간을 초과했습니다.', 504) from exc
        except (OSError, ValueError) as exc:
            self._stop()
            self.model_state = 'failed'; self.last_error_code = 'process_failed'
            raise ForecastError('process_failed', '예측 프로세스 응답을 확인할 수 없습니다.') from exc
        except ForecastError as exc:
            self.last_error_code = exc.code
            if exc.code == 'process_failed':
                self._stop()
            if exc.code != 'invalid_history':
                self.model_state = 'failed'
            elif self.model_state == 'loading':
                self.model_state = 'not_loaded'
            raise

    def close(self):
        with self.lock:
            self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
        self._stop()


broker = ForecastBroker()
atexit.register(broker.close)
