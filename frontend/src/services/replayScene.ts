import type {ReplayEvent,ReplayFrame,ReplayResult} from './api';
import {replayFrameAt} from './replayTiming';

export type FloorPoint=[number,number,number];
export interface ReplayActor {trackId:number;x:number;y:number;floor:FloorPoint;heading:number;moving:boolean;uncertain:boolean;zone:string|null}
export interface ReplayScene {frame:number|null;actors:ReplayActor[];events:ReplayEvent[];unidentified:number;zoneNames:string[]}
export interface ReplayFilters {route?:[string,string]|null;zone?:string|null}

// Dimensionless, reversible image-plane mapping. No homography/metres/geography.
export function imageToFloor(x:number,y:number,[width,height]:[number,number]):FloorPoint {
  const scale=20/width;
  return [(x-width/2)*scale,0,(y-height/2)*scale];
}
export function floorToImage([x,,z]:FloorPoint,[width,height]:[number,number]):[number,number] {
  return [x*width/20+width/2,z*width/20+height/2];
}

export function selectReplayScene(data:ReplayResult,frame:ReplayFrame|undefined,time:number,filters:ReplayFilters={}):ReplayScene {
  const routeIds=filters.route?new Set(data.route_track_ids?data.route_track_ids.find(r=>r.from_zone===filters.route![0]&&r.to_zone===filters.route![1])?.track_ids??[]:data.events.filter(e=>e.from_zone===filters.route![0]&&e.to_zone===filters.route![1]).map(e=>e.track_id)):null;
  const matchesZone=(zone:string|null)=>filters.zone===undefined||zone===(filters.zone===null?'':filters.zone);
  const counts=new Map<number,number>();
  for(const p of frame?.points??[])if(p.track_id!==null)counts.set(p.track_id,(counts.get(p.track_id)??0)+1);
  // Only a directly adjacent observed frame may provide heading. Never bridge a gap.
  const previous=frame&&frame.frame>0?replayFrameAt(data,(frame.frame-1)/data.fps):undefined;
  const actors:ReplayActor[]=[];let unidentified=0;
  for(const p of frame?.points??[]){
    if(p.track_id===null||!Number.isSafeInteger(p.track_id)||p.track_id<=0||counts.get(p.track_id)!==1||!Number.isFinite(p.x+p.y)){unidentified++;continue;}
    if(!matchesZone(p.zone)||routeIds&&!routeIds.has(p.track_id))continue;
    const uncertain=p.zone===null||data.diagnostics.some(d=>d.track_id===p.track_id&&(d.frame===frame!.frame||d.resumed_frame===frame!.frame));
    const prev=previous?.points.find(q=>q.track_id===p.track_id);
    const moving=!uncertain&&!!prev&&Math.hypot(p.x-prev.x,p.y-prev.y)>=.5;
    actors.push({trackId:p.track_id,x:p.x,y:p.y,floor:imageToFloor(p.x,p.y,data.video_resolution),
      heading:moving?Math.atan2(p.x-prev!.x,p.y-prev!.y):0,moving,uncertain,zone:p.zone});
  }
  const events=data.events.filter(e=>e.timestamp_ms<=time*1000&&e.timestamp_ms>time*1000-1000
    &&(!filters.route||e.from_zone===filters.route[0]&&e.to_zone===filters.route[1])
    &&(filters.zone===undefined||e.from_zone===(filters.zone===null?'':filters.zone)||e.to_zone===(filters.zone===null?'':filters.zone)));
  return {frame:frame?.frame??null,actors,events,unidentified,zoneNames:data.roi_snapshot.zones.filter(z=>z.visible!==false).map(z=>z.name)};
}

export function threePreference(hints:{width:number;cores?:number;memory?:number}):string|null {
  if(hints.width<=640)return '모바일 화면에서는 2D를 기본으로 권장합니다.';
  if((hints.cores!==undefined&&hints.cores<=2)||(hints.memory!==undefined&&hints.memory<=2))return '기기 자원이 적어 2D를 권장합니다.';
  return null;
}
