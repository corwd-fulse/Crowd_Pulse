import type {DetectionResult,DetectionFrame} from './api';
// Tracker enumerates frames from zero and stores floor(frame / sourceFPS * 1000).
export function detectionAt(data:DetectionResult,seconds:number):DetectionFrame|undefined {
  if(!Number.isFinite(seconds)||seconds<0||seconds>=data.frame_count/data.fps)return;
  const frame=Math.floor(seconds*data.fps+1e-6);
  let lo=0,hi=data.frames.length-1;
  while(lo<=hi){const mid=(lo+hi)>>1;const value=data.frames[mid];if(value.frame<frame)lo=mid+1;else if(value.frame>frame)hi=mid-1;else return Math.abs(value.t_ms-frame/data.fps*1000)<=1.01?value:undefined;}
}
