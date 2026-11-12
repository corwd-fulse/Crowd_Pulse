import {useEffect, useState} from 'react';
import axios from 'axios';
import {getVisionResultBundle} from '../services/api';
import type {AnalysisJobResponse, VisionResultBundle} from '../services/api';

export type ResultState =
  | {status: 'idle'}
  | {status: 'loading'}
  | {status: 'success'; data: VisionResultBundle}
  | {status: 'error'; message: string};

export function useVisionResult(job: AnalysisJobResponse | null) {
  const [attempt, setAttempt] = useState(0);
  const [stored, setStored] = useState<{jobId: string; attempt: number; state: ResultState} | null>(null);
  const jobId = job?.id;
  const jobStatus = job?.status;
  useEffect(() => {
    if (!jobId || jobStatus !== 'succeeded') return;
    const controller = new AbortController();
    getVisionResultBundle(jobId, controller.signal).then((data) => {
      if (!controller.signal.aborted) setStored({jobId, attempt, state: {status: 'success', data}});
    }).catch((error: unknown) => {
      if (controller.signal.aborted) return;
      const message = axios.isAxiosError(error) && error.response?.status === 404
        ? '분석 완료 상태이지만 저장된 결과가 없습니다.'
        : '분석 결과를 불러오지 못했습니다. 연결 상태를 확인하고 다시 시도하세요.';
      setStored({jobId, attempt, state: {status: 'error', message}});
    });
    return () => controller.abort();
  }, [jobId, jobStatus, attempt]);
  const state: ResultState = !jobId || jobStatus !== 'succeeded' ? {status: 'idle'}
    : stored?.jobId === jobId && stored.attempt === attempt ? stored.state : {status: 'loading'};
  return {state, retry: () => setAttempt((value) => value + 1)};
}
