import {act, fireEvent, render, screen} from '@testing-library/react';
import {afterEach, beforeEach, expect, it, vi} from 'vitest';
import {useState} from 'react';
import type {ReactElement} from 'react';
import {MemoryRouter,useLocation} from 'react-router-dom';
import Dashboard from '../src/pages/DashBoard';
import AnalysisResults from '../src/components/dashboard/AnalysisResults';
import {useAnalysisJobPolling} from '../src/hooks/useAnalysisJobPolling';
import {getAnalysisJobs, getAnalysisJobsByVideo, getDashboardSummary, getVisionResultBundle, getVisionResult,getLatestCompletedJob,getVideoDetails,getAnalysisJob,getVisionObservations} from '../src/services/api';
import type {AnalysisJobResponse, VisionResultBundle} from '../src/services/api';
import sample from './fixtures/visionBundle.json';
import jobSample from './fixtures/analysisJob.json';

vi.mock('../src/services/api', async (importOriginal) => ({
  ...await importOriginal<typeof import('../src/services/api')>(),
  getAnalysisJobs:vi.fn(),getAnalysisJobsByVideo:vi.fn(),getVisionResultBundle:vi.fn(),getDashboardSummary:vi.fn(),getVisionResult:vi.fn(),
  getLatestCompletedJob:vi.fn(),getVideoDetails:vi.fn(),getAnalysisJob:vi.fn(),
  getVisionObservations:vi.fn(),
}));
vi.mock('../src/components/common/ServiceStatus', () => ({default:() => <div>API 상태 표시</div>}));
vi.mock('recharts', async (importOriginal) => {
  const react = await import('react');
  return {...await importOriginal<typeof import('recharts')>(),
    ResponsiveContainer: ({children}: {children: ReactElement<{width: number; height: number}>}) => react.cloneElement(children, {width:800,height:240}),
  };
});

const job=jobSample as AnalysisJobResponse;
beforeEach(() => {
  vi.mocked(getAnalysisJobs).mockReset().mockResolvedValue([job]);
  vi.mocked(getAnalysisJobsByVideo).mockReset().mockResolvedValue([job]);
  vi.mocked(getVisionResultBundle).mockReset().mockResolvedValue(sample as VisionResultBundle);
  vi.mocked(getVisionResult).mockReset().mockResolvedValue((sample as VisionResultBundle).summary);
  vi.mocked(getLatestCompletedJob).mockReset().mockResolvedValue(job);
  vi.mocked(getVideoDetails).mockReset().mockImplementation(async id=>({id,filename:'fixture.mp4'}));
  vi.mocked(getAnalysisJob).mockReset().mockResolvedValue(job);
  vi.mocked(getVisionObservations).mockReset().mockResolvedValue(sample as VisionResultBundle);
  vi.mocked(getDashboardSummary).mockClear();
});
afterEach(() => {vi.useRealTimers();});

function Location() {return <output aria-label="현재 주소">{useLocation().search}</output>;}

it('defaults to latest completion outside the recent list and persists it in the URL',async()=>{
  vi.mocked(getAnalysisJobs).mockResolvedValue([{...job,id:'new-queued',status:'queued'}]);
  render(<MemoryRouter initialEntries={['/dashboard?view=all']}><Dashboard/><Location/></MemoryRouter>);
  await screen.findByText('테스트 분석 결과');
  expect((screen.getByRole('combobox',{name:'조회 작업'}) as HTMLSelectElement).value).toBe(job.id);
  expect(screen.getByLabelText('현재 주소').textContent).toContain(`job=${job.id}`);
  expect(screen.getByLabelText('현재 주소').textContent).toContain('view=all');
  expect(getLatestCompletedJob).toHaveBeenCalled();
});

it('honors a valid queued URL and keeps an explicit user selection across list refresh',async()=>{
  const queued={...job,id:'queued-job',video_id:'queued-video',status:'queued' as const};
  vi.mocked(getAnalysisJobs).mockResolvedValue([queued,job]);
  render(<MemoryRouter initialEntries={[`/dashboard?job=${job.id}`]}><Dashboard/><Location/></MemoryRouter>);
  await screen.findByText('테스트 분석 결과');
  fireEvent.change(screen.getByRole('combobox',{name:'조회 작업'}),{target:{value:queued.id}});
  await screen.findByText(/미분석 또는 분석 미완료/);
  fireEvent.click(screen.getByRole('button',{name:'작업 목록 새로고침'}));
  expect((screen.getByRole('combobox',{name:'조회 작업'}) as HTMLSelectElement).value).toBe(queued.id);
  expect(screen.getByLabelText('현재 주소').textContent).toContain(`job=${queued.id}`);
  expect(getLatestCompletedJob).not.toHaveBeenCalled();
});

it('does not auto-select queued work when no completed video exists',async()=>{
  vi.mocked(getAnalysisJobs).mockResolvedValue([{...job,id:'queued-only',status:'queued'}]);
  vi.mocked(getLatestCompletedJob).mockResolvedValue(null);
  render(<MemoryRouter initialEntries={['/dashboard']}><Dashboard/></MemoryRouter>);
  await screen.findByText(/분석 완료된 영상이 없습니다/);
  expect(screen.queryByLabelText('대시보드 서버 영상')).toBeNull();
  expect(getVisionResultBundle).not.toHaveBeenCalled();
});

it('loads a real selected job, preserves upload controls, and makes mock panels explicitly opt-in', async () => {
  const {container}=render(<MemoryRouter initialEntries={[`/dashboard?job=${job.id}`]}><Dashboard/></MemoryRouter>);
  await screen.findByText('테스트 분석 결과');
  expect(screen.getByText('영상 업로드')).toBeTruthy();
  expect(container.querySelector('[data-source="mock"]')).toBeNull();
  expect(getDashboardSummary).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button',{name:'Mock 시연 보기 (실제 분석과 별도)'}));
  expect(container.querySelector('[data-source="mock"]')).not.toBeNull();
  expect(screen.getByText(/시연 전용 샘플 데이터/)).toBeTruthy();
});

it('reuses the 5-second poll, fetches results after succeeded, and stops terminal polling', async () => {
  vi.useFakeTimers();
  function Harness() {
    const [selected,setSelected]=useState<AnalysisJobResponse>({...job,status:'running'});
    useAnalysisJobPolling(selected.video_id,selected.status==='running',(items) => setSelected(items[0]),() => {});
    return <AnalysisResults job={selected}/>;
  }
  render(<Harness/>);
  expect(getVisionResultBundle).not.toHaveBeenCalled();
  await act(async () => {await vi.advanceTimersByTimeAsync(4999);});
  expect(getAnalysisJobsByVideo).not.toHaveBeenCalled();
  await act(async () => {await vi.advanceTimersByTimeAsync(1);});
  expect(getAnalysisJobsByVideo).toHaveBeenCalledTimes(1);
  expect(getVisionResultBundle).toHaveBeenCalledTimes(1);
  expect(screen.getByText('테스트 분석 결과')).toBeTruthy();
  await act(async () => {await vi.advanceTimersByTimeAsync(10000);});
  expect(getAnalysisJobsByVideo).toHaveBeenCalledTimes(1);
});

it('does not poll hidden tabs or continue after unmount', async () => {
  vi.useFakeTimers();
  const hidden=vi.spyOn(document,'visibilityState','get').mockReturnValue('hidden');
  function Harness() {
    useAnalysisJobPolling(job.video_id,true,() => {},() => {});
    return null;
  }
  const view=render(<Harness/>);
  await act(async () => {await vi.advanceTimersByTimeAsync(5000);});
  expect(getAnalysisJobsByVideo).not.toHaveBeenCalled();
  hidden.mockReturnValue('visible');
  view.unmount();
  await act(async () => {await vi.advanceTimersByTimeAsync(5000);});
  expect(getAnalysisJobsByVideo).not.toHaveBeenCalled();
});

it('ignores an old poll immediately when video selection version changes', async () => {
  vi.useFakeTimers();
  const version={current:0};
  const receive=vi.fn();
  let resolve!: (jobs: AnalysisJobResponse[]) => void;
  vi.mocked(getAnalysisJobsByVideo).mockImplementationOnce(() => new Promise((done) => {resolve=done;}));
  function Harness() {
    useAnalysisJobPolling(job.video_id,true,receive,() => {},undefined,version);
    return null;
  }
  render(<Harness/>);
  await act(async () => {await vi.advanceTimersByTimeAsync(5000);});
  version.current++;
  await act(async () => resolve([job]));
  expect(receive).not.toHaveBeenCalled();
});
