import {fireEvent,render,screen} from '@testing-library/react';
import {beforeEach,expect,it,vi} from 'vitest';
import DetectionOverlay from '../src/components/dashboard/DetectionOverlay';
import {getDetections} from '../src/services/api';
vi.mock('../src/services/api',async original=>({...await original<typeof import('../src/services/api')>(),getDetections:vi.fn()}));
const data={analysis_job_id:'job',video_id:'video',fps:25,frame_count:3,video_resolution:[1920,1080] as [number,number],frames:[{frame:0,t_ms:0,boxes:[{track_id:7,x1:10,y1:20,x2:100,y2:200,confidence:.9}]}]};
beforeEach(()=>vi.mocked(getDetections).mockReset().mockResolvedValue(data));
it('shows stored boxes without ROI and clears them at a frame with no detections',async()=>{
  const video={current:document.createElement('video')};
  const {container}=render(<DetectionOverlay jobId="job" videoId="video" video={video} resolution={[1920,1080]} show/>);
  await screen.findByText('ID 7');
  const box=container.querySelector('rect')!;
  expect(box.getAttribute('x')).toBe('10');expect(box.getAttribute('width')).toBe('90');
  video.current.currentTime=.04;fireEvent(video.current,new Event('seeked'));
  expect(container.querySelector('rect')).toBeNull();
});
it('distinguishes missing CSV from load error and never invents boxes',async()=>{
  vi.mocked(getDetections).mockResolvedValueOnce(null).mockRejectedValueOnce(new Error('corrupt'));
  const props={videoId:'video',video:{current:document.createElement('video')},resolution:[1920,1080] as [number,number],show:true};
  const {rerender,container}=render(<DetectionOverlay {...props} jobId="missing"/>);
  await screen.findByText('이 작업의 탐지 CSV가 없습니다.');
  rerender(<DetectionOverlay {...props} jobId="error"/>);
  await screen.findByText('탐지 CSV 로딩·검증 오류입니다.');
  expect(container.querySelector('rect')).toBeNull();
});
it('hides old results immediately when switching jobs and respects the visibility toggle',async()=>{
  const props={videoId:'video',video:{current:document.createElement('video')},resolution:[1920,1080] as [number,number]};
  const {rerender,container}=render(<DetectionOverlay {...props} jobId="job" show/>);
  await screen.findByText('ID 7');
  rerender(<DetectionOverlay {...props} jobId="job" show={false}/>);
  expect(container.querySelector('rect')).toBeNull();
  rerender(<DetectionOverlay {...props} jobId={null} show/>);
  expect(screen.getByText('완료된 분석 작업이 없습니다.')).toBeTruthy();
  expect(container.querySelector('rect')).toBeNull();
});
