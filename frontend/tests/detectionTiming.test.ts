import {expect,it} from 'vitest';
import {detectionAt} from '../src/services/detectionTiming';
import type {DetectionResult} from '../src/services/api';
const data:DetectionResult={analysis_job_id:'job',video_id:'video',fps:25,frame_count:4,video_resolution:[1920,1080],frames:[{frame:0,t_ms:0,boxes:[]},{frame:1,t_ms:40,boxes:[]},{frame:3,t_ms:120,boxes:[]}]};
it('uses original fps, zero-based frame and tracker milliseconds at exact boundaries',()=>{
  expect(detectionAt(data,0)?.frame).toBe(0);
  expect(detectionAt(data,.039)?.frame).toBe(0);
  expect(detectionAt(data,.04)?.frame).toBe(1);
  expect(detectionAt(data,.12)?.frame).toBe(3);
});
it('never holds a previous box over an empty frame or after end of video',()=>{
  expect(detectionAt(data,.08)).toBeUndefined();
  expect(detectionAt(data,.16)).toBeUndefined();
  expect(detectionAt(data,-1)).toBeUndefined();
  expect(detectionAt(data,NaN)).toBeUndefined();
});
it('handles fractional source fps without rounding it to playback throughput',()=>{
  const fractional={...data,fps:29.97,frames:[{frame:3,t_ms:100,boxes:[]}]};
  expect(detectionAt(fractional,3/29.97)?.frame).toBe(3);
  expect(detectionAt({...fractional,frames:[{frame:3,t_ms:200,boxes:[]}]},3/29.97)).toBeUndefined();
});
