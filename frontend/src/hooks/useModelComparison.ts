import {useEffect, useState} from 'react';
import {getModelComparison} from '../services/api';
import type {ModelComparisonResponse} from '../services/api';

export type ModelComparisonState =
  | {status: 'idle'}
  | {status: 'loading'}
  | {status: 'not_run'} // DM-Count 분석 결과가 없는 경우 — 에러가 아니라 정상적인 "미실행" 상태
  | {status: 'success'; data: ModelComparisonResponse}
  | {status: 'error'; message: string};

// jobId가 완료된(succeeded) 분석 작업일 때만 호출하는 쪽(AnalysisResults)이 보장한다 —
// useVisionResult와 같은 분리: job 상태 분기는 호출부가, 결과 조회 상태는 이 훅이 맡는다.
export function useModelComparison(jobId: string | null) {
  const [attempt, setAttempt] = useState(0);
  const [stored, setStored] = useState<{jobId: string; attempt: number; state: ModelComparisonState} | null>(null);
  useEffect(() => {
    if (!jobId) return;
    const controller = new AbortController();
    getModelComparison(jobId, controller.signal).then((data) => {
      if (controller.signal.aborted) return;
      setStored({jobId, attempt, state: data === null ? {status: 'not_run'} : {status: 'success', data}});
    }).catch(() => {
      if (controller.signal.aborted) return;
      setStored({jobId, attempt, state: {status: 'error', message: '모델 비교 결과를 불러오지 못했습니다. 연결 상태를 확인하고 다시 시도하세요.'}});
    });
    return () => controller.abort();
  }, [jobId, attempt]);
  const state: ModelComparisonState = !jobId ? {status: 'idle'}
    : stored?.jobId === jobId && stored.attempt === attempt ? stored.state : {status: 'loading'};
  return {state, retry: () => setAttempt((value) => value + 1)};
}
