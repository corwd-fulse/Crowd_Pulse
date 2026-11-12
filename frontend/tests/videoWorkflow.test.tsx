import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import {beforeEach,expect,it,vi} from 'vitest';
import {MemoryRouter} from 'react-router-dom';
import VideoManagement from '../src/pages/VideoManagement';
import {getVideos,getAnalysisJobsByVideo,getVideoRoi,getAnalysisCapabilities,createAnalysisJob,getDetections,getVisionObservations} from '../src/services/api';
import type {VideoListItem,AnalysisJobResponse} from '../src/services/api';
import jobJson from './fixtures/analysisJob.json';
import bundle from './fixtures/visionBundle.json';
vi.mock('../src/services/api',async original=>({...await original<typeof import('../src/services/api')>(),getVideos:vi.fn(),getAnalysisJobsByVideo:vi.fn(),getVideoRoi:vi.fn(),getAnalysisCapabilities:vi.fn(),createAnalysisJob:vi.fn(),getDetections:vi.fn(),getVisionObservations:vi.fn()}));
beforeEach(()=>{
  vi.mocked(getVideos).mockReset().mockResolvedValue([{id:'new-video',filename:'new.mp4',status:'uploaded',file_size:100,uploaded_at:'2026-11-12T00:00:00Z'} as VideoListItem]);
  vi.mocked(getAnalysisJobsByVideo).mockReset().mockResolvedValue([]);
  vi.mocked(getVideoRoi).mockReset().mockResolvedValue(null);
  vi.mocked(getAnalysisCapabilities).mockReset().mockResolvedValue({test_analysis_enabled:false,execution:'external_worker',auto_start_worker:false});
  vi.mocked(createAnalysisJob).mockReset();
  vi.mocked(getDetections).mockReset().mockImplementation(async id=>({analysis_job_id:id,video_id:'new-video',fps:25,frame_count:404,video_resolution:[1920,1080],frames:[]}));
  vi.mocked(getVisionObservations).mockReset().mockImplementation(async id=>({summary:{...bundle.summary,analysis_job_id:id,video_id:'new-video'},metrics:[]}));
});
async function open() {
  render(<MemoryRouter><VideoManagement/></MemoryRouter>);
  fireEvent.click(await screen.findByRole('button',{name:/new.mp4/}));
  await waitFor(()=>expect(screen.queryByText('분석 작업 조회 중…')).toBeNull());
}
it('explains an unanalysed video and places analysis controls before the preview',async()=>{
  await open();
  expect(screen.getByText('아직 분석하지 않은 영상입니다')).toBeTruthy();
  expect((screen.getByRole('button',{name:'분석 요청'}) as HTMLButtonElement).disabled).toBe(true);
  await screen.findByText(/서버 ROI 미등록 — 미저장 초안/);
  expect(screen.getByText(/별도 Worker가 실행되어야 시작/)).toBeTruthy();
});
it('exposes request-time missing ROI on a queued job without changing it',async()=>{
  const queued={...jobJson,id:'queued-new',video_id:'new-video',status:'queued',roi_configured:false} as AnalysisJobResponse;
  vi.mocked(getAnalysisJobsByVideo).mockResolvedValue([queued]);await open();
  await screen.findByRole('heading',{name:'분석 대기 중'});
  expect(screen.getByText(/이 작업은 요청 당시 유효한 ROI가 없었습니다/)).toBeTruthy();
  expect((screen.getByRole('button',{name:'분석 요청됨'}) as HTMLButtonElement).disabled).toBe(true);
});
it('shows AI running state',async()=>{
  vi.mocked(getAnalysisJobsByVideo).mockResolvedValue([{...jobJson,id:'running',video_id:'new-video',status:'running'} as AnalysisJobResponse]);await open();
  await screen.findByText('AI 분석 중');
  const select=screen.getByRole('combobox',{name:'분석 범위'}) as HTMLSelectElement;
  expect(select.value).toBe('full');
  expect(select.disabled).toBe(true);
  expect(screen.getByRole('option',{name:'FULL — 정밀 분석'})).toBeTruthy();
  expect(screen.getByText(/현재 작업의 분석 범위를 표시합니다/)).toBeTruthy();
});
it('shows the failed job cause and server-disabled test mode',async()=>{
  vi.mocked(getAnalysisJobsByVideo).mockResolvedValue([{...jobJson,id:'failed',video_id:'new-video',status:'failed',error_message:'ROI_NOT_APPROVED'} as AnalysisJobResponse]);await open();
  await screen.findByText('분석 실패: ROI_NOT_APPROVED');
  expect(screen.getByText(/현재 API 서버에서 비활성화됨/)).toBeTruthy();
  expect((screen.getByRole('option',{name:'테스트 — 서버 허용 필요'}) as HTMLOptionElement).disabled).toBe(true);
});

it('selects a new request immediately, hides old results while queued and reconnects after completion',async()=>{
  const old={...jobJson,id:'old-done',video_id:'new-video'} as AnalysisJobResponse;
  const queued={...old,id:'new-request',status:'queued'} as AnalysisJobResponse;
  vi.mocked(getVideoRoi).mockResolvedValue({video_id:'new-video',roi_snapshot:{...bundle.summary.roi_snapshot,status:'approved',approved_for_analysis:true}});
  vi.mocked(getAnalysisJobsByVideo).mockResolvedValueOnce([old]).mockResolvedValueOnce([queued,old]).mockResolvedValueOnce([{...queued,status:'succeeded'},old]);
  vi.mocked(createAnalysisJob).mockResolvedValue(queued);
  await open();
  const video=screen.getByLabelText('ROI 원본 영상');
  Object.defineProperties(video,{videoWidth:{value:1920},videoHeight:{value:1080}});fireEvent.loadedMetadata(video);
  await waitFor(()=>expect((screen.getByRole('button',{name:'분석 요청'}) as HTMLButtonElement).disabled).toBe(false));
  fireEvent.click(screen.getByRole('button',{name:'분석 요청'}));
  await screen.findByRole('heading',{name:'분석 대기 중'});
  expect(document.querySelector('[data-source=zone-observations]')).toBeNull();
  await waitFor(()=>expect((screen.getByRole('button',{name:'분석 이력 새로고침'}) as HTMLButtonElement).disabled).toBe(false));
  fireEvent.click(screen.getByRole('button',{name:'분석 이력 새로고침'}));
  await waitFor(()=>expect((screen.getByRole('combobox',{name:'탐지 결과 작업'}) as HTMLSelectElement).value).toBe(queued.id));
  await waitFor(()=>expect(getDetections).toHaveBeenCalledWith(queued.id,expect.any(AbortSignal)));
  expect(createAnalysisJob).toHaveBeenCalledWith('new-video','production','fast');
});
