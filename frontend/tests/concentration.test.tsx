import {useState} from 'react';
import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import {beforeEach,expect,it,vi} from 'vitest';
import ConcentrationPanel from '../src/components/dashboard/ConcentrationPanel';
import ConcentrationMapLayer from '../src/components/dashboard/ConcentrationMapLayer';
import {emptyVenue} from '../src/services/venueControl';
import type {VenueConfiguration} from '../src/services/venueControl';
import type {AnalysisJobResponse} from '../src/services/api';
import {getAnalysisJobs} from '../src/services/api';
import {estimateConcentration,getOperatorEvents,incidentBrief,recordOperatorEvent} from '../src/services/concentration';
import type {ConcentrationResult} from '../src/services/concentration';

vi.mock('../src/services/api',async original=>({...await original<typeof import('../src/services/api')>(),getAnalysisJobs:vi.fn(),getVisionResult:vi.fn().mockResolvedValue({analysis_job_id:'job',video_id:'video',roi_snapshot:{zones:[{name:'A'}]},video_resolution:[100,100]})}));
vi.mock('../src/services/concentration',async original=>({...await original<typeof import('../src/services/concentration')>(),estimateConcentration:vi.fn(),getOperatorEvents:vi.fn(),recordOperatorEvent:vi.fn()}));
const job={id:'job',video_id:'video',status:'succeeded',analysis_mode:'test'} as AnalysisJobResponse;
const venue:VenueConfiguration={...emptyVenue,zones:[{id:'center',name:'CENTER',polygon:[[0,0],[100,0],[100,100]],color:'#44b6c8',mapping:null,note:'',visible:true,capacity:null}],center_zone_id:'center'};
const result={run_id:'run',revision:1,venue_id:'main',venue_name:'QA venue',registered_address:null,center_zone_id:'center',created_at:'2026-11-17T00:00:00Z',model_version:'central-flow-constant-rate-v1',method:'conditional',mode:'independent',observation_kind:'recorded_video_observation',observed_at:null,reference_offset_seconds:25,window_seconds:30,cameras:[],routes:[],observed_center_count:null,center_total_coverage_verified:false,aggregation_allowed:false,aggregation_reasons:['independent_videos_not_aggregated'],state:'미분석',state_basis:[],threshold_basis:'administrator_proposal_not_safety_standard',increasing_corridors:0,estimates:[{horizon_seconds:10,target_at:null,target_offset_seconds:35,expected_arrivals:null,expected_departures:null,expected_center_count:null,inward_pressure_per_minute:null,confidence:'unavailable',reasons:['center_total_unavailable'],validation_status:'not_validated'}],priority_corridors:[]} as ConcentrationResult;
beforeEach(()=>{vi.mocked(getAnalysisJobs).mockResolvedValue([job]);vi.mocked(estimateConcentration).mockResolvedValue(result);vi.mocked(getOperatorEvents).mockResolvedValue([]);vi.mocked(recordOperatorEvent).mockResolvedValue({});});

it('supports four or more independent camera mappings without defaulting to synchronization',async()=>{
  let last=venue;
  function Wrapper(){const [config,setConfig]=useState(venue);last=config;return <ConcentrationPanel config={config} revision={1} dirty time={25} job={job} onChange={setConfig} onResult={()=>{}} onPlaceCamera={()=>{}}/>;}
  render(<Wrapper/>);
  await waitFor(()=>expect(getAnalysisJobs).toHaveBeenCalled());
  for(let i=0;i<5;i++)fireEvent.click(screen.getByRole('button',{name:'카메라 추가'}));
  expect(last.cameras).toHaveLength(5);
  expect(new Set(last.cameras?.map(c=>c.camera_id)).size).toBe(5);
  expect(last.cameras?.every(c=>!c.clock_verified&&!c.position_confirmed)).toBe(true);
  expect(screen.getByRole('button',{name:'중앙 집중 추정 조회'}).hasAttribute('disabled')).toBe(true);
});

it('shows unknown center, saves explicit operator action, and does not dial or send a report',async()=>{
  const onResult=vi.fn();
  render(<ConcentrationPanel config={venue} revision={1} dirty={false} time={25} job={job} onChange={()=>{}} onResult={onResult} onPlaceCamera={()=>{}}/>);
  fireEvent.click(screen.getByRole('button',{name:'중앙 집중 추정 조회'}));
  await screen.findByLabelText('중앙 집중 추정 결과');
  expect(screen.getAllByText(/독립 영상은 합산하지 않음/).length).toBeGreaterThan(0);
  const brief=screen.getByLabelText('신고 참고 정보') as HTMLTextAreaElement;
  expect(brief.value).toContain('미등록');expect(brief.value).toContain('미관측');
  expect(document.querySelector('a[href^="tel:"]')).toBeNull();
  expect(recordOperatorEvent).not.toHaveBeenCalled();
  fireEvent.change(screen.getByLabelText('관제 확인 담당자'),{target:{value:'Operator'}});
  fireEvent.change(screen.getByLabelText('현장 확인 상태'),{target:{value:'checking'}});
  fireEvent.change(screen.getByLabelText('관리자 대응 메모'),{target:{value:'영상 확인 필요'}});
  fireEvent.click(screen.getByRole('button',{name:'관리자 확인·메모 기록'}));
  await waitFor(()=>expect(recordOperatorEvent).toHaveBeenCalledWith(expect.objectContaining({run_id:'run',verification:'checking',note:'영상 확인 필요'}),'main'));
});

it('invalidates results when playback time changes',async()=>{
  const props={config:venue,revision:1,dirty:false,time:25,job,onChange:()=>{},onResult:vi.fn(),onPlaceCamera:()=>{}};
  const {rerender}=render(<ConcentrationPanel {...props}/>);
  fireEvent.click(screen.getByRole('button',{name:'중앙 집중 추정 조회'}));
  await screen.findByLabelText('중앙 집중 추정 결과');
  rerender(<ConcentrationPanel {...props} time={26}/>);
  expect(screen.queryByLabelText('중앙 집중 추정 결과')).toBeNull();
});

it('keeps virtual scenario and missing address explicit in report support',()=>{
  const brief=incidentBrief({...result,mode:'scenario'},'CENTER','현장 미확인','');
  expect(brief).toContain('가상 시나리오');expect(brief).toContain('실제 동시 관측 아님');
  expect(brief).toContain('실제 주소 확인 필요');expect(brief).toContain('기다리지 말고');
});

it('aligns the master recording time to the registered capture clock',async()=>{
  const config:VenueConfiguration={...venue,integration_mode:'synchronized',cameras:[{camera_id:'cam',name:'Camera',registered_camera_id:null,point:[0,0],position_confirmed:false,bearing_deg:null,zone_id:'center',job_id:'job',roi_name:'A',coverage:[],full_zone_coverage:false,capture_start:'2026-11-17T00:00:00+09:00',clock_verified:true,clock_evidence:'clock fixture',scenario_offset_s:0}]};
  render(<ConcentrationPanel config={config} revision={1} dirty={false} time={25} job={job} onChange={()=>{}} onResult={()=>{}} onPlaceCamera={()=>{}}/>);
  fireEvent.click(screen.getByRole('button',{name:'중앙 집중 추정 조회'}));
  await waitFor(()=>expect(estimateConcentration).toHaveBeenCalledWith(1,25,'2026-11-16T15:00:25.000Z',expect.any(AbortSignal),'main'));
  expect((screen.getByLabelText('공통 촬영 기준 시각') as HTMLInputElement).readOnly).toBe(true);
});

it('draws separated inward and outward gate arrows without inventing people markers',()=>{
  const config:VenueConfiguration={...venue,zones:[...venue.zones,{...venue.zones[0],id:'a',name:'A',polygon:[[200,200],[300,200],[300,300]]}]};
  const data={...result,routes:[{corridor_id:'g',name:'Gate',camera_id:'cam',from_zone_id:'a',to_zone_id:'center',toward_center:2,away_from_center:1,net:1,toward_change:null,window_seconds:10,quality:'usable_low_confidence',reason:null,arrival_basis:'unknown',arrival_ratio:null,travel_seconds:null}]};
  const {container}=render(<svg><ConcentrationMapLayer config={config} result={data}/></svg>);
  const arrows=container.querySelectorAll('[data-central-corridor="g"] path');
  expect(arrows.length).toBe(2);expect(arrows[0].getAttribute('d')).not.toBe(arrows[1].getAttribute('d'));
  expect(container.querySelectorAll('circle').length).toBe(0);
});
