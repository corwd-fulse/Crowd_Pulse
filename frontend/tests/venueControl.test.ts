import {expect,it} from 'vitest';
import {observedRoutes,zoneObservation} from '../src/services/venueControl';
import type {VenueZone} from '../src/services/venueControl';
import type {ReplayResult} from '../src/services/api';

const zones:VenueZone[]=['A','B','C'].map((name,i)=>({id:name,name,polygon:[[i*250,0],[i*250+200,0],[i*250+200,200]],color:'#44b6c8',visible:true,capacity:null,mapping:{job_id:'job',roi_name:name},note:''}));
const event=(from:string,to:string,t:number,kind:'boundary'|'zone_path'='boundary')=>({event_id:from+to+t,kind,track_id:1,segment_id:1,from_zone:from,to_zone:to,frame:t*25,timestamp_ms:t*1000,confidence:.9,from_position:[0,0] as [number,number],to_position:[1,1] as [number,number],via_outside:kind==='zone_path'});
const data={analysis_job_id:'job',fps:25,frame_count:500,frames:[{frame:250,t_ms:10000,observed:true,flow_complete:true,points:[],zones:zones.map(z=>({zone:z.name,count:2,inflow:1,outflow:1,net_inflow:0,confirmed_inflow:1,confirmed_outflow:1}))}],events:[event('A','B',1),event('B','A',2),event('A','B',3),event('B','C',4)],observed_od:[{from_zone:'A',to_zone:'C',tracks:1}]} as ReplayResult;

it('keeps both directions, applies the time window, and never turns OD into a route',()=>{
  expect(observedRoutes(zones,'job',data,10,10).map(r=>[r.from.name,r.to.name,r.count])).toEqual([['A','B',2],['B','A',1],['B','C',1]]);
  expect(observedRoutes(zones,'job',data,4,1).map(r=>[r.from.name,r.to.name,r.count])).toEqual([['B','C',1]]);
  expect(observedRoutes(zones,'different',data,10,10)).toEqual([]);
});
it('does not double count outside-mediated path events as boundary inflow',()=>{
  const outside={...data,events:[event('A','',1),event('','B',2),event('A','B',2,'zone_path')]};
  expect(zoneObservation(zones[1],'job',outside,null,10,10).inflow).toBe(1);
  expect(observedRoutes(zones,'job',outside,10,10)[0].count).toBe(1);
});
it('preserves unobserved, disconnected, and unmapped states instead of inventing zero',()=>{
  expect(zoneObservation(zones[0],'job',data,null,10.04,10).count).toBeNull();
  expect(zoneObservation({...zones[0],mapping:null},'job',data,null,10,10).count).toBeNull();
  const incomplete={...data,frames:[{...data.frames[0],flow_complete:false}]};
  const result=zoneObservation(zones[0],'job',incomplete,null,10,10);
  expect(result.count).toBe(2);expect(result.inflow).toBeNull();expect(result.outflow).toBeNull();expect(result.score).toBeUndefined();
});
