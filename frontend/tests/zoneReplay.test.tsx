import {createRef} from 'react';
import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import {beforeEach,expect,it,vi} from 'vitest';
import ZoneReplay from '../src/components/dashboard/ZoneReplay';
import {getReplay,getVisionZoneMetrics,getVisionBottleneckScores} from '../src/services/api';
import type {ReplayResult} from '../src/services/api';
import {replayFrameAt,canvasPoint} from '../src/services/replayTiming';
import {nextZoneName} from '../src/services/roiGeometry';
vi.mock('../src/services/api',async original=>({...await original<typeof import('../src/services/api')>(),getReplay:vi.fn(),getVisionZoneMetrics:vi.fn(),getVisionBottleneckScores:vi.fn()}));
const data:ReplayResult={version:'v1',analysis_job_id:'job',video_id:'video',fps:25,frame_count:100,video_resolution:[1920,1080],mode:'observed_replay_not_prediction',
  roi_snapshot:{status:'candidate',approved_for_analysis:false,video_resolution:[1920,1080],zones:[]},
  frames:[{frame:50,t_ms:2000,observed:true,flow_complete:false,points:[{track_id:1,x:100,y:100,zone:'A',confidence:.9}],zones:[{zone:'A',count:1,inflow:null,outflow:null,net_inflow:null,confirmed_inflow:0,confirmed_outflow:0}]}],
  events:[],routes:[{from_zone:'A',to_zone:'B',transitions:1}],observed_od:[],od_note:'observed',diagnostics:[],coverage:{verified_frame_count:100,complete_frame_index:true,incomplete_track_ids:1,unresolved_paths:1}};
beforeEach(()=>{
  vi.mocked(getReplay).mockResolvedValue(structuredClone(data));
  vi.mocked(getVisionZoneMetrics).mockResolvedValue({items:[],limit:1000,offset:0,direction_presentation:{} as never});
  vi.mocked(getVisionBottleneckScores).mockResolvedValue({items:[],limit:1000,offset:0});
  vi.spyOn(HTMLCanvasElement.prototype,'getContext').mockReturnValue(null);
});
it('uses exact observed frames and original-coordinate canvas mapping',()=>{
  expect(replayFrameAt(data,2)?.frame).toBe(50);
  expect(replayFrameAt(data,2.04)).toBeUndefined();
  expect(replayFrameAt(data,4)).toBeUndefined();
  expect(canvasPoint(960,540,[1920,1080],480,270)).toEqual([240,135]);
  expect(nextZoneName([{name:'A',polygon:[]},{name:'C',polygon:[]}])).toBe('B');
});
it('shares video time, seeking and playback speed without inventing missing markers',async()=>{
  const video=document.createElement('video');const ref=createRef<HTMLVideoElement>();ref.current=video;
  const {container}=render(<ZoneReplay jobId="job" videoId="video" video={ref} resolution={[1920,1080]} summary={null}/>);
  await screen.findByLabelText('리플레이 타임라인');
  video.currentTime=2;fireEvent(video,new Event('seeked'));
  await waitFor(()=>expect(container.querySelector('canvas')?.getAttribute('data-replay-frame')).toBe('50'));
  fireEvent.change(screen.getByLabelText('리플레이 배속'),{target:{value:'2'}});expect(video.playbackRate).toBe(2);
  fireEvent.change(screen.getByLabelText('리플레이 타임라인'),{target:{value:'2.04'}});
  expect(video.currentTime).toBeCloseTo(2.04);
  expect(container.querySelector('canvas')?.getAttribute('data-replay-frame')).toBe('');
  expect(screen.getByText(/총 유입\/유출은 미분석/)).toBeTruthy();
});
it('rejects a replay from another video',async()=>{
  vi.mocked(getReplay).mockResolvedValue({...data,video_id:'other'});
  const ref=createRef<HTMLVideoElement>();ref.current=document.createElement('video');
  render(<ZoneReplay jobId="job" videoId="video" video={ref} resolution={[1920,1080]} summary={null}/>);
  await screen.findByRole('alert');expect(screen.queryByRole('slider')).toBeNull();
});
it('retains 2D and its current frame when WebGL2 is unavailable',async()=>{
  const ref=createRef<HTMLVideoElement>();ref.current=document.createElement('video');
  const {container}=render(<ZoneReplay jobId="job" videoId="video" video={ref} resolution={[1920,1080]} summary={null}/>);
  await screen.findByLabelText('리플레이 타임라인');
  ref.current.currentTime=2;fireEvent(ref.current,new Event('seeked'));
  fireEvent.click(screen.getByRole('button',{name:'3D 보기'}));
  expect(screen.getByText(/WebGL2를 사용할 수 없어/)).toBeTruthy();
  expect(container.querySelector('canvas')?.getAttribute('data-replay-frame')).toBe('50');
});
it('uses decoded frame time during playback while preserving exact paused seeks',async()=>{
  const ref=createRef<HTMLVideoElement>();const video=document.createElement('video');ref.current=video;
  let callback:VideoFrameRequestCallback|undefined;
  video.requestVideoFrameCallback=vi.fn(fn=>{callback=fn;return 1;});
  video.cancelVideoFrameCallback=vi.fn();
  Object.defineProperty(video,'paused',{configurable:true,value:false});
  const {container}=render(<ZoneReplay jobId="job" videoId="video" video={ref} resolution={[1920,1080]} summary={null}/>);
  await screen.findByLabelText('리플레이 타임라인');
  video.currentTime=2;fireEvent(video,new Event('seeked'));
  await waitFor(()=>expect(container.querySelector('canvas')?.getAttribute('data-replay-frame')).toBe('50'));
  video.currentTime=2.04;fireEvent(video,new Event('timeupdate'));
  expect(container.querySelector('canvas')?.getAttribute('data-replay-frame')).toBe('50');
  Object.defineProperty(video,'paused',{configurable:true,value:true});
  fireEvent(video,new Event('seeked'));
  await waitFor(()=>expect(container.querySelector('canvas')?.getAttribute('data-replay-frame')).toBe(''));
  expect(callback).toBeTypeOf('function');
});
