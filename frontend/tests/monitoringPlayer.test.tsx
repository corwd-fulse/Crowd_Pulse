import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import {afterEach,beforeEach,expect,it,vi} from 'vitest';
import CameraViewer from '../src/components/dashboard/CameraViewer';
import {getDetections,getVisionResult,uploadVideo,getVideoDetails,getVideoRoi,createAnalysisJob,getVisionObservations} from '../src/services/api';
import type {AnalysisJobResponse,VisionResultBundle} from '../src/services/api';
import jobJson from './fixtures/analysisJob.json';
import bundle from './fixtures/visionBundle.json';
vi.mock('../src/services/api',async original=>({...await original<typeof import('../src/services/api')>(),getDetections:vi.fn(),getVisionResult:vi.fn(),uploadVideo:vi.fn(),getVideoDetails:vi.fn(),getVideoRoi:vi.fn(),createAnalysisJob:vi.fn(),getVisionObservations:vi.fn()}));
const job=jobJson as AnalysisJobResponse;
const detections={analysis_job_id:job.id,video_id:job.video_id,fps:25,frame_count:404,video_resolution:[1920,1080] as [number,number],frames:[{frame:50,t_ms:2000,boxes:[{track_id:7,x1:10,y1:20,x2:100,y2:200,confidence:.9}]}]};
beforeEach(()=>{
  vi.mocked(getDetections).mockReset().mockResolvedValue(detections);
  vi.mocked(getVisionResult).mockReset().mockResolvedValue((bundle as VisionResultBundle).summary);
  vi.mocked(uploadVideo).mockReset();
  vi.mocked(getVideoDetails).mockReset().mockImplementation(async id=>({id,filename:'fixture.mp4'}));
  vi.mocked(getVideoRoi).mockReset().mockResolvedValue(null);
  vi.mocked(createAnalysisJob).mockReset();
  vi.mocked(getVisionObservations).mockReset().mockResolvedValue(bundle as VisionResultBundle);
});
afterEach(()=>vi.unstubAllGlobals());
function metadata() {
  const video=screen.getByLabelText('대시보드 서버 영상') as HTMLVideoElement;
  Object.defineProperties(video,{videoWidth:{value:1920},videoHeight:{value:1080}});fireEvent.loadedMetadata(video);
  return video;
}
it('uses the selected job server video and common detection overlay on the dashboard',async()=>{
  const {container}=render(<CameraViewer trackedJob={job}/>);
  const video=metadata();
  expect(video.getAttribute('src')).toContain(`/api/videos/${job.video_id}/stream`);
  expect(screen.getByText('서버 영상 재생')).toBeTruthy();
  await waitFor(()=>expect(getDetections).toHaveBeenCalledWith(job.id,expect.any(AbortSignal)));
  video.currentTime=2;fireEvent(video,new Event('seeked'));
  await screen.findByText('ID 7');
  expect(container.querySelector('svg[data-frame] rect')?.getAttribute('width')).toBe('90');
  for(const name of ['원본 영상','YOLO 탐지','ROI 구역'])expect(screen.getByRole('checkbox',{name})).toBeTruthy();
  fireEvent.click(screen.getByRole('checkbox',{name:'YOLO 탐지'}));
  expect(container.querySelector('svg[data-frame]')).toBeNull();
});
it('discards prior boxes and switches server source immediately for an unfinished job',async()=>{
  const {container,rerender}=render(<CameraViewer trackedJob={job}/>);
  const v=metadata();v.currentTime=2;fireEvent(v,new Event('seeked'));await screen.findByText('ID 7');
  rerender(<CameraViewer trackedJob={{...job,id:'queued-job',video_id:'different-video',status:'queued'}}/>);
  expect(screen.getByLabelText('대시보드 서버 영상').getAttribute('src')).toContain('/different-video/stream');
  expect(container.querySelector('svg[data-frame]')).toBeNull();
  expect(screen.getByText(/미분석 또는 분석 미완료/)).toBeTruthy();
  expect(getDetections).toHaveBeenCalledTimes(1);
});
it('keeps a selected local file separate even after upload succeeds',async()=>{
  vi.stubGlobal('URL',class extends URL {static createObjectURL(){return 'blob:local-only';} static revokeObjectURL(){}});
  vi.mocked(uploadVideo).mockResolvedValue({video_id:'new-upload',filename:'local.mp4',saved_filename:'new.mp4',file_size:5,status:'uploaded',data_source:'user_upload'});
  const {container}=render(<CameraViewer trackedJob={job}/>);metadata();
  await waitFor(()=>expect(getDetections).toHaveBeenCalledTimes(1));
  fireEvent.change(container.querySelector('input[type=file]')!,{target:{files:[new File(['local'],'local.mp4',{type:'video/mp4'})]}});
  await screen.findByText(/로컬 파일 재생 · 서버 업로드 완료/);
  expect(screen.getByLabelText('대시보드 로컬 영상').getAttribute('src')).toBe('blob:local-only');
  expect(container.querySelector('svg[data-frame]')).toBeNull();
  fireEvent.click(screen.getByRole('button',{name:'분석 요청'}));
  await screen.findByText(/분석 요청 전 영상 관리에서 ROI를 저장하고 승인을 확인하세요/);
  expect(createAnalysisJob).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button',{name:'선택 작업의 서버 영상 보기'}));
  expect(screen.getByLabelText('대시보드 서버 영상').getAttribute('src')).toContain(job.video_id);
});
it('rejects detections associated with another video',async()=>{
  vi.mocked(getDetections).mockResolvedValue({...detections,video_id:'wrong-video'});
  const {container}=render(<CameraViewer trackedJob={job}/>);metadata();
  await screen.findByText('탐지 CSV 로딩·검증 오류입니다.');
  expect(container.querySelector('svg[data-frame] rect')).toBeNull();
});
