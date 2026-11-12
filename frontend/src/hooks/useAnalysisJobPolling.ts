import {useEffect, useEffectEvent, useRef} from 'react';
import type {RefObject} from 'react';
import {getAnalysisJobsByVideo} from '../services/api';
import type {AnalysisJobResponse} from '../services/api';

export const ANALYSIS_POLLING_INTERVAL = 5000;

export function useAnalysisJobPolling(
  videoId: string | null | undefined,
  enabled: boolean,
  onJobs: (jobs: AnalysisJobResponse[]) => void,
  onError: (error: unknown) => void,
  busyRef?: RefObject<boolean>,
  selectionVersionRef?: RefObject<number>,
) {
  const localBusy = useRef(false);
  const busy = busyRef ?? localBusy;
  const receive = useEffectEvent(onJobs);
  const fail = useEffectEvent(onError);
  useEffect(() => {
    if (!videoId || !enabled) return;
    let cancelled = false;
    let fetching = false;
    const version = selectionVersionRef?.current;
    const current = () => !cancelled && (!selectionVersionRef || selectionVersionRef.current === version);
    const timer = window.setInterval(async () => {
      if (document.visibilityState === 'hidden' || fetching || busy.current) return;
      fetching = true;
      busy.current = true;
      try {
        const jobs = await getAnalysisJobsByVideo(videoId);
        if (current()) receive(jobs);
      } catch (error) {
        if (current()) fail(error);
      } finally {
        fetching = false;
        busy.current = false;
      }
    }, ANALYSIS_POLLING_INTERVAL);
    return () => {cancelled = true; window.clearInterval(timer);};
  }, [videoId, enabled, busy, selectionVersionRef]);
}
