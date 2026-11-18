import type {ReplayFrame,ReplayResult} from './api';

export function replayFrameAt(data:ReplayResult,time:number):ReplayFrame|undefined {
  if(!Number.isFinite(time)||time<0||time>=data.frame_count/data.fps)return;
  const frame=Math.floor(time*data.fps+1e-6);
  let low=0,high=data.frames.length-1;
  while(low<=high){const mid=(low+high)>>1;const row=data.frames[mid];if(row.frame===frame)return row;if(row.frame<frame)low=mid+1;else high=mid-1;}
  // Missing frames intentionally produce no marker, never nearest-neighbour motion.
}

export function canvasPoint(x:number,y:number,resolution:[number,number],width:number,height:number):[number,number] {
  return [x/resolution[0]*width,y/resolution[1]*height];
}
