import {expect,it} from 'vitest';
import type {ReplayResult} from '../src/services/api';
import {selectReplayScene,imageToFloor,floorToImage,threePreference} from '../src/services/replayScene';
import {replayFrameAt} from '../src/services/replayTiming';
import sample from './fixtures/replay3DSample.json';
const data=sample as unknown as ReplayResult;

for(const frame of [50,200,300])it(`keeps actual IDs and reversible coordinates at frame ${frame}`,()=>{
  const observed=replayFrameAt(data,frame/data.fps)!;
  const scene=selectReplayScene(data,observed,frame/data.fps);
  expect(scene.actors.map(a=>a.trackId)).toEqual(observed.points.map(p=>p.track_id));
  for(const actor of scene.actors){
    const expected=observed.points.find(p=>p.track_id===actor.trackId)!;
    expect(actor.x).toBe(expected.x);expect(actor.y).toBe(expected.y);
    const [x,y]=floorToImage(actor.floor,data.video_resolution);
    expect(x).toBeCloseTo(expected.x,9);expect(y).toBeCloseTo(expected.y,9);
  }
});
it('keeps both route directions and exact event IDs at observed timestamps',()=>{
  for(const route of data.routes){
    for(const event of data.events.filter(e=>e.from_zone===route.from_zone&&e.to_zone===route.to_zone)){
      const scene=selectReplayScene(data,replayFrameAt(data,event.frame/data.fps),event.timestamp_ms/1000,{route:[route.from_zone,route.to_zone]});
      expect(scene.events.map(e=>e.event_id)).toContain(event.event_id);
      expect(scene.events.every(e=>e.from_zone===route.from_zone&&e.to_zone===route.to_zone)).toBe(true);
    }
  }
});
it('shows no actors at a missing frame and never bridges heading across gaps',()=>{
  const single={...data,frames:[data.frames[0],{...data.frames[0],frame:52,t_ms:2080}]} as ReplayResult;
  expect(selectReplayScene(single,replayFrameAt(single,2.04),2.04).actors).toHaveLength(0);
  expect(selectReplayScene(single,single.frames[1],2.08).actors.every(a=>!a.moving)).toBe(true);
});
it('filters the shared actor list by snapshot zone and rejects unknown/duplicate IDs',()=>{
  const frame=replayFrameAt(data,2)!;
  expect(selectReplayScene(data,frame,2,{zone:'walkway_far'}).actors.every(a=>a.zone==='walkway_far')).toBe(true);
  const broken={...frame,points:[{...frame.points[0],track_id:null},frame.points[1],frame.points[1]]};
  const scene=selectReplayScene(data,broken,2);
  expect(scene.actors).toHaveLength(0);expect(scene.unidentified).toBe(3);
});
it('uses dimensionless image mapping and conservative device hints',()=>{
  expect(imageToFloor(960,540,[1920,1080])).toEqual([0,0,0]);
  expect(threePreference({width:390,cores:8})).toContain('모바일');
  expect(threePreference({width:1440,cores:2})).toContain('자원');
});
