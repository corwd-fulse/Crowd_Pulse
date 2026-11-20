import {api} from './api';
import type {ReplayResult, VisionResultBundle} from './api';
import {replayFrameAt} from './replayTiming';
import type {Point} from './roiGeometry';
import {bottleneckReason} from './visionPresentation';
import type {ConcentrationPolicy,Corridor,VenueCamera} from './concentration';

export interface VenueZone {id:string;name:string;polygon:Point[];color:string;visible:boolean;capacity:number|null;mapping:{job_id:string;roi_name:string}|null;note:string}
export interface VenueMarker {id:string;name:string;kind:'entrance'|'facility'|'camera';point:Point}
export interface VenueLink {id:string;name:string;from_zone_id:string;to_zone_id:string;kind:'walkway'|'entrance'|'stairs'|'bottleneck';bidirectional:boolean;camera_ids:string[]}
export interface VenueConfiguration {links?:VenueLink[];venue_type?:string;name:string;coordinate_system:'venue_normalized_1000';image:string|null;width:number;height:number;zones:VenueZone[];markers:VenueMarker[];mapping_roi_snapshots?:Record<string,unknown>;camera_sources?:Record<string,unknown>;address?:string;center_zone_id?:string|null;integration_mode?:'independent'|'scenario'|'synchronized';cameras?:VenueCamera[];corridors?:Corridor[];concentration_policy?:ConcentrationPolicy;target_policies?:Record<string,ConcentrationPolicy>}
export interface VenueRevision {venue_id:string;revision:number;created_at:string;configuration:VenueConfiguration}
export const emptyVenue:VenueConfiguration={name:'행사장 관제',coordinate_system:'venue_normalized_1000',image:null,width:1600,height:1000,zones:[],markers:[]};
export const getVenue=(signal?:AbortSignal,venueId='main')=>api.get<VenueRevision>(`/api/venues/${encodeURIComponent(venueId)}`,{signal}).then(r=>r.data);
export const getVenueHistory=(venueId='main')=>api.get<{revision:number;created_at:string}[]>(`/api/venues/${encodeURIComponent(venueId)}/revisions`).then(r=>r.data);
export const getVenueRevision=(revision:number,venueId='main')=>api.get<VenueRevision>(`/api/venues/${encodeURIComponent(venueId)}/revisions/${revision}`).then(r=>r.data);
export function saveVenue(configuration:VenueConfiguration,base_revision:number,venueId='main'){
  const {mapping_roi_snapshots:_,camera_sources:_sources,...editable}=configuration;
  return api.post<VenueRevision>(`/api/venues/${encodeURIComponent(venueId)}/revisions`,{configuration:editable,base_revision}).then(r=>r.data);
}
export const center=(p:Point[]):Point=>[p.reduce((n,v)=>n+v[0],0)/p.length,p.reduce((n,v)=>n+v[1],0)/p.length];
export function removeVenueZone(config:VenueConfiguration,id:string):VenueConfiguration{
  const cameras=(config.cameras??[]).filter(c=>c.zone_id!==id),ids=new Set(cameras.map(c=>c.camera_id));
  const policy=config.concentration_policy;
  return {...config,links:config.links?.filter(link=>link.from_zone_id!==id&&link.to_zone_id!==id).map(link=>({...link,camera_ids:link.camera_ids.filter(c=>ids.has(c))})),target_policies:Object.fromEntries(Object.entries(config.target_policies??{}).filter(([key])=>key!==id)),zones:config.zones.filter(z=>z.id!==id),cameras,
    center_zone_id:config.center_zone_id===id?null:config.center_zone_id,
    corridors:(config.corridors??[]).filter(g=>g.from_zone_id!==id&&g.to_zone_id!==id&&ids.has(g.camera_id)),
    concentration_policy:policy?{...policy,center_observer_camera_id:policy.center_observer_camera_id&&ids.has(policy.center_observer_camera_id)?policy.center_observer_camera_id:null,
      center_boundary_coverage_complete:config.center_zone_id===id?false:policy.center_boundary_coverage_complete}:undefined};
}
export const value=(n:number|null|undefined)=>n==null?'미관측':n.toLocaleString('ko-KR',{maximumFractionDigits:1});

export function zoneObservation(zone:VenueZone,jobId:string|undefined,replay:ReplayResult|null,bundle:VisionResultBundle|null,time:number,windowSeconds:number){
  const mapped=!!jobId&&zone.mapping?.job_id===jobId;
  const roi=zone.mapping?.roi_name;
  const frame=mapped&&replay?.analysis_job_id===jobId?replayFrameAt(replay,time):undefined;
  const row=frame?.observed?frame.zones.find(z=>z.zone===roi):undefined;
  const count=row?.count??null;
  const events=mapped?replay?.events.filter(e=>e.kind==='boundary'&&e.timestamp_ms>Math.max(-1,(time-windowSeconds)*1000)&&e.timestamp_ms<=time*1000)??[]:[];
  const inflow=count!==null&&frame?.flow_complete?events.filter(e=>e.to_zone===roi).length:null;
  const outflow=count!==null&&frame?.flow_complete?events.filter(e=>e.from_zone===roi).length:null;
  const metric=mapped?bundle?.metrics.find(m=>m.region_key===roi&&Math.floor(m.video_offset_ms/1000)===Math.floor(time)):undefined;
  const status=mapped?bundle?.summary.bottleneck_status?.find(s=>s.zone===roi&&s.window_start===Math.floor(time/10)*10):undefined;
  const unavailable=status?.status==='unanalyzed'||Object.values(status?.missing_metrics??{}).some(v=>v.length)||Object.values(status?.previous_missing_metrics??{}).some(v=>v.length);
  const score=count!==null&&mapped&&!unavailable&&bundle?.summary.analysis_profile!=='fast'?bundle?.scores.find(s=>s.region_key===roi&&s.window_start_ms===Math.floor(time/10)*10000):undefined;
  // Existing backend severity is authoritative. No invented density thresholds.
  const color=count===null||!score?'#738398':score.severity==='high'?'#f97316':score.severity==='medium'?'#eabf50':'#39b88b';
  return {count,inflow,outflow,net:inflow===null||outflow===null?null:inflow-outflow,metric,score,color,
    label:count===null?'미관측':!score?'혼잡 지표 미분석':score.severity==='high'?'병목 지표 높음':score.severity==='medium'?'병목 지표 주의':'병목 지표 낮음',
    reason:!mapped?'현재 작업과 수동 매핑 필요':count===null?'해당 영상 프레임 관측 없음':unavailable&&status?bottleneckReason(status):!score?'병목 점수 미분석':metric?.low_confidence?'Optical Flow 저신뢰':'관측 기록 있음'};
}

export function observedRoutes(zones:VenueZone[],jobId:string|undefined,replay:ReplayResult|null,time:number,seconds:number){
  if(!jobId||replay?.analysis_job_id!==jobId)return [];
  const names=new Map(zones.filter(z=>z.mapping?.job_id===jobId).map(z=>[z.mapping!.roi_name,z]));
  const counts=new Map<string,{from:VenueZone;to:VenueZone;count:number}>();
  for(const e of replay?.events??[]){
    if(e.timestamp_ms<=Math.max(-1,(time-seconds)*1000)||e.timestamp_ms>time*1000)continue;
    const from=names.get(e.from_zone),to=names.get(e.to_zone);
    if(!from||!to||from.id===to.id)continue;
    const key=JSON.stringify([from.id,to.id]);
    const row=counts.get(key)??{from,to,count:0};row.count++;counts.set(key,row);
  }
  return [...counts.values()];
}

// SVG is decoded as an image and rasterized: no executable SVG is persisted.
export async function readFloorplan(file:File):Promise<Pick<VenueConfiguration,'image'|'width'|'height'>>{
  if(file.size>5*1024*1024||!['image/png','image/jpeg','image/svg+xml'].includes(file.type))throw new Error('PNG/JPG/SVG 5MiB 이하 파일을 선택하세요.');
  const url=URL.createObjectURL(file);
  try{
    const img=new Image();img.src=url;await img.decode();
    if(!img.naturalWidth||!img.naturalHeight||img.naturalWidth*img.naturalHeight>24000000)throw new Error('평면도 해상도가 너무 큽니다 (최대 24MP).');
    const c=document.createElement('canvas');c.width=img.naturalWidth;c.height=img.naturalHeight;c.getContext('2d')!.drawImage(img,0,0);
    const image=c.toDataURL('image/png');if(image.length>7000000)throw new Error('변환한 평면도가 5MiB를 초과합니다. 이미지 크기를 줄여 주세요.');
    return {image,width:c.width,height:c.height};
  }finally{URL.revokeObjectURL(url);}
}
