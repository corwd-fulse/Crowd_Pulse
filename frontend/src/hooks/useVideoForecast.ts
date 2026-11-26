import {useEffect, useState} from 'react';
import {getVideoForecast} from '../services/api';
import type {VideoForecastResponse, VideoForecastSource} from '../services/api';

export type VideoForecastState =
  | {status: 'loading'}
  | {status: 'not_run'} // 이 job에 시계열 예측을 돌리지 않음 — 에러가 아님
  | {status: 'success'; data: VideoForecastResponse}
  | {status: 'error'; message: string};

// useModelComparison과 같은 구조: 요청 키(job·출처·예측 시간·재시도)가 바뀌면 새로 조회한다.
export function useVideoForecast(jobId: string, source: VideoForecastSource, horizon: number) {
  const [attempt, setAttempt] = useState(0);
  const key = `${jobId}|${source}|${horizon}|${attempt}`;
  const [stored, setStored] = useState<{key: string; state: VideoForecastState} | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    getVideoForecast(jobId, source, horizon, controller.signal).then((data) => {
      if (controller.signal.aborted) return;
      setStored({key, state: data === null ? {status: 'not_run'} : {status: 'success', data}});
    }).catch(() => {
      if (controller.signal.aborted) return;
      setStored({key, state: {status: 'error', message: '시계열 예측 결과를 불러오지 못했습니다. 연결 상태를 확인하고 다시 시도하세요.'}});
    });
    return () => controller.abort();
  }, [jobId, source, horizon, key]);
  const state: VideoForecastState = stored?.key === key ? stored.state : {status: 'loading'};
  return {state, retry: () => setAttempt((value) => value + 1)};
}
