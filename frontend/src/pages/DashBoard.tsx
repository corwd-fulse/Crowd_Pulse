import {useEffect, useState} from 'react';
import {useSearchParams} from 'react-router-dom';
import CameraViewer from '../components/dashboard/CameraViewer';
import RiskPanel from '../components/dashboard/RiskPanel';
import CrowdChart from '../components/dashboard/CrowdChart';
import FacilityMap from '../components/dashboard/FacilityMap';
import AnalysisResults from '../components/dashboard/AnalysisResults';
import ServerStatus from '../components/common/ServiceStatus';
import {getAnalysisJob, getAnalysisJobs, getLatestCompletedJob} from '../services/api';
import type {AnalysisJobResponse} from '../services/api';
import {useAnalysisJobPolling} from '../hooks/useAnalysisJobPolling';

const statusLabels: Record<AnalysisJobResponse['status'], string> = {
  queued: '분석 대기', running: '분석 중', succeeded: '분석 완료', failed: '분석 실패', cancelled: '취소',
};

export default function Dashboard() {
  const [search, setSearch] = useSearchParams();
  const requestedId = search.get('job');
  const [jobs, setJobs] = useState<AnalysisJobResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [refresh, setRefresh] = useState(0);
  const [showDemo, setShowDemo] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      const [list,latest] = await Promise.all([getAnalysisJobs(controller.signal),requestedId?Promise.resolve(null):getLatestCompletedJob(controller.signal)]);
      if (requestedId && !list.some((job) => job.id === requestedId)) {
        list.unshift(await getAnalysisJob(requestedId, controller.signal));
      }
      if(latest&&!list.some(job=>job.id===latest.id))list.unshift(latest);
      return {list,latest};
    }
    load().then(({list,latest}) => {
      if (!controller.signal.aborted) {
        setJobs(list);setError('');
        if(!requestedId&&latest)setSearch(previous=>{const next=new URLSearchParams(previous);next.set('job',latest.id);return next;},{replace:true});
      }
    }).catch(() => {
      if (!controller.signal.aborted) setError('분석 작업 목록을 불러오지 못했습니다. 연결 상태와 작업 ID를 확인하세요.');
    }).finally(() => {if (!controller.signal.aborted) setLoading(false);});
    return () => controller.abort();
  }, [requestedId, refresh, setSearch]);
  const selected = requestedId ? jobs.find((job) => job.id === requestedId) ?? null : null;
  useAnalysisJobPolling(selected?.video_id, selected?.status === 'queued' || selected?.status === 'running', (incoming) => {
    setJobs((previous) => previous.map((job) => incoming.find((updated) => updated.id === job.id) ?? job));
    setError('');
  }, () => setError('분석 상태 자동 조회에 실패했습니다. 다음 5초 주기에 다시 확인합니다.'));
  const onJobCreated = (job: AnalysisJobResponse) => {
    setJobs((previous) => [job, ...previous.filter((item) => item.id !== job.id)]);
    setSearch({job: job.id});
  };
  return (
    <div className="command-dashboard space-y-5">
      <ServerStatus />
      <section className="rounded-lg border border-line bg-surface p-5" aria-label="분석 작업 선택">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div><h1 className="text-lg font-bold text-ink">영상 AI 분석 대시보드</h1><p className="mt-1 text-xs text-muted">기본 선택: 가장 최근 완료 작업 · 최근 작업 최대 100개와 선택한 과거 작업 표시</p></div>
          <button type="button" disabled={loading} onClick={() => {setLoading(true); setRefresh((value) => value + 1);}} className="rounded-sm border border-line px-3 py-2 text-xs text-ink-soft disabled:opacity-40">작업 목록 새로고침</button>
        </div>
        <label className="mt-4 block text-xs text-ink-soft">조회 작업<select aria-label="조회 작업" value={selected?.id ?? ''} onChange={(event) => setSearch({job:event.target.value})} className="mt-2 w-full rounded-sm border border-line bg-surface p-2 text-xs"><option value="" disabled>{loading ? '작업 목록 로딩 중…' : '작업을 선택하세요'}</option>{jobs.map((job) => <option key={job.id} value={job.id}>{job.analysis_profile==='fast'?'FAST':'FULL'} · {statusLabels[job.status]} · {job.analysis_mode === 'test' ? '테스트' : '운영'} · {new Date(job.created_at).toLocaleString('ko-KR')} · {job.id}</option>)}</select></label>
        {error && <p role="alert" className="mt-3 text-sm text-danger">{error}</p>}
        {!loading&&!error&&!requestedId&&<p role="status" className="mt-3 text-sm text-muted">분석 완료된 영상이 없습니다. 대기·진행 중인 작업은 목록에서 직접 선택할 수 있습니다.</p>}
      </section>
      <section className="grid grid-cols-1 items-stretch gap-5 xl:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]">
        <CameraViewer onJobCreated={onJobCreated} trackedJob={selected} />
        <div className="rounded-lg border border-line bg-surface p-5 text-sm text-ink-soft"><h2 className="font-bold text-ink">선택한 분석 작업</h2><p className="mt-3">{selected ? statusLabels[selected.status] : '작업 선택 대기'}</p>{selected && <p className="mt-2 break-all text-xs text-muted">{selected.id}</p>}<p className="mt-4 text-xs text-muted">아래 지표는 선택한 작업의 실제 저장 결과입니다. 분석 대기·진행 중에는 5초마다 상태를 확인하고 완료 시 결과를 조회합니다.</p><p className="mt-3 text-xs text-muted">새 영상 업로드와 분석 요청은 왼쪽 화면에서 계속 사용할 수 있습니다. 원본 영상과 결과의 작업 ID를 확인하세요.</p></div>
      </section>
      {loading && !selected ? <p role="status" className="text-sm text-muted">분석 작업 목록을 불러오는 중…</p> : <AnalysisResults job={selected} />}
      <section className="rounded-lg border border-dashed border-line p-4" data-source="mock-demo-toggle">
        <button type="button" aria-expanded={showDemo} onClick={() => setShowDemo((value) => !value)} className="text-xs text-muted">{showDemo ? 'Mock 시연 닫기' : 'Mock 시연 보기 (실제 분석과 별도)'}</button>
        {showDemo && <div data-source="mock" className="mt-4 space-y-5"><p className="text-sm font-semibold text-warn">시연 전용 샘플 데이터 — 아래 지도·위험·예측 수치는 실제 AI 결과가 아닙니다.</p><FacilityMap/><RiskPanel/><CrowdChart/></div>}
      </section>
    </div>
  );
}
