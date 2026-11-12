import {fireEvent,render,screen} from '@testing-library/react';
import {beforeEach,expect,it,vi} from 'vitest';
import VideoZoneCounts from '../src/components/dashboard/VideoZoneCounts';
import {getVisionObservations} from '../src/services/api';
import type {VisionResultBundle} from '../src/services/api';
import json from './fixtures/visionBundle.json';
vi.mock('../src/services/api',async original=>({...await original<typeof import('../src/services/api')>(),getVisionObservations:vi.fn()}));
const bundle=json as VisionResultBundle;
beforeEach(()=>vi.mocked(getVisionObservations).mockReset().mockResolvedValue(bundle));
it('shows video-relative per-second decimal means, changes buckets on seek, and preserves zero',async()=>{
  const row=bundle.metrics[0];
  vi.mocked(getVisionObservations).mockResolvedValue({...bundle,metrics:[{...row,video_offset_ms:0,region_key:'near',mean_person_count:3.9},{...row,video_offset_ms:1000,region_key:'near',mean_person_count:0}]});
  const video={current:document.createElement('video')};
  render(<VideoZoneCounts jobId={bundle.summary.analysis_job_id} videoId={bundle.summary.video_id} video={video}/>);
  await screen.findByText('near: 3.9명');
  video.current.currentTime=1.8;fireEvent(video.current,new Event('seeked'));
  expect(screen.getByText('near: 0명')).toBeTruthy();
  video.current.currentTime=3;fireEvent(video.current,new Event('timeupdate'));
  expect(screen.getByText('near: 데이터 없음')).toBeTruthy();
  expect(screen.getByText(/현재 프레임의 탐지 박스 수나 고유 인원 수가 아닙니다/)).toBeTruthy();
});
it('does not request or display counts before a completed job exists',()=>{
  const {container}=render(<VideoZoneCounts jobId={null} videoId="new-video" video={{current:null}}/>);
  expect(getVisionObservations).not.toHaveBeenCalled();expect(container.textContent).toBe('');
});
it('rejects cross-video results and removes prior counts immediately when the job changes',async()=>{
  const props={videoId:bundle.summary.video_id,video:{current:document.createElement('video')}};
  const {rerender}=render(<VideoZoneCounts {...props} jobId={bundle.summary.analysis_job_id}/>);
  await screen.findByText(/walkway_far: /);
  vi.mocked(getVisionObservations).mockResolvedValue({...bundle,summary:{...bundle.summary,analysis_job_id:'new-job',video_id:'wrong-video'}});
  rerender(<VideoZoneCounts {...props} jobId="new-job"/>);
  expect(screen.queryByText(/walkway_far: /)).toBeNull();
  await screen.findByRole('alert');
});
