import {useEffect,useRef,useState} from 'react';
import type {AnalysisJobResponse,VisionResultSummary} from '../../services/api';
import {api,getAnalysisJobs,getVisionResult} from '../../services/api';
import type {Point} from '../../services/roiGeometry';
import {screenToVideo} from '../../services/roiGeometry';
import type {VenueConfiguration} from '../../services/venueControl';
import {defaultPolicy,estimateConcentration,evaluateConcentration,getOperatorEvents,incidentBrief,reasonLabel,recordOperatorEvent} from '../../services/concentration';
import type {ConcentrationPolicy,ConcentrationResult,Corridor,Evaluation,OperatorRecord,VenueCamera} from '../../services/concentration';
import './concentration.css';

const number=(n:number|null|undefined)=>n==null?'미관측·미분석':n.toLocaleString('ko-KR',{maximumFractionDigits:2});
const verificationLabels:Record<string,string>={unconfirmed:'현장 미확인',checking:'현장 확인 중',field_verified:'현장 확인 기록됨'};
const timestamp=(s:string)=>new Date(s).toLocaleString('ko-KR',{timeZone:'Asia/Seoul'});

export default function ConcentrationPanel({config,revision,dirty,time,job,onChange,onResult,onPlaceCamera,venueId='main'}:{venueId?:string;
  config:VenueConfiguration;revision:number;dirty:boolean;time:number;job:AnalysisJobResponse|null;
  onChange:(next:VenueConfiguration)=>void;onResult:(r:ConcentrationResult|null)=>void;onPlaceCamera:(id:string)=>void;
}){
  const cameras=config.cameras??[],corridors=config.corridors??[],policy={...defaultPolicy,...config.concentration_policy};
  const [jobs,setJobs]=useState<AnalysisJobResponse[]>([]),[selected,setSelected]=useState(''),[gateId,setGateId]=useState('');
  const [summary,setSummary]=useState<VisionResultSummary|null>(null),[message,setMessage]=useState(''),[busy,setBusy]=useState(false);
  const [result,setResult]=useState<ConcentrationResult|null>(null),[evaluation,setEvaluation]=useState<Evaluation|null>(null),[at,setAt]=useState('');
  const [operator,setOperator]=useState(''),[verification,setVerification]=useState('unconfirmed'),[action,setAction]=useState('review'),[note,setNote]=useState('');
  const [events,setEvents]=useState<OperatorRecord[]>([]),[confirmedState,setConfirmedState]=useState('현장 미확인');
  const current=useRef({config,revision,dirty,time});current.current={config,revision,dirty,time};
  const callback=useRef(onResult);callback.current=onResult;
  const controller=useRef<AbortController|null>(null);
  const activeRun=useRef<string|null>(null);activeRun.current=result?.run_id??null;
  const camera=cameras.find(c=>c.camera_id===selected),gate=corridors.find(g=>g.corridor_id===gateId&&g.camera_id===selected);
  const mode=config.integration_mode??'independent';
  const referenceCamera=cameras.find(c=>c.job_id===job?.id&&c.clock_verified&&c.capture_start);
  const referenceMillis=referenceCamera?.capture_start?new Date(referenceCamera.capture_start).getTime()+time*1000:NaN;
  const commonAt=Number.isFinite(referenceMillis)?new Date(referenceMillis).toISOString():at;
  useEffect(()=>{const c=new AbortController();getAnalysisJobs(c.signal).then(async list=>{if(venueId!=='main'){const sources=await api.get<{id:string}[]>(`/api/capture/venues/${encodeURIComponent(venueId)}/sources`,{signal:c.signal});const allowed=new Set(sources.data.map(s=>s.id));const existing=new Set((config.cameras??[]).map(c=>c.job_id));list=list.filter(j=>allowed.has(j.video_id)||existing.has(j.id));}if(!c.signal.aborted)setJobs(job&&!list.some(j=>j.id===job.id)?[job,...list]:list);}).catch(()=>{if(!c.signal.aborted){setJobs(job?[job]:[]);setMessage('분석 작업 목록 조회 실패 — 현재 작업만 연결 가능');}});return()=>c.abort();},[job?.id,venueId]);
  useEffect(()=>{setSummary(null);if(!camera?.job_id)return;const c=new AbortController();getVisionResult(camera.job_id,c.signal).then(data=>{if(data.analysis_job_id!==camera.job_id)throw new Error('작업 불일치');if(!c.signal.aborted)setSummary(data);}).catch(()=>{if(!c.signal.aborted)setMessage('카메라의 저장된 ROI 조회 실패');});return()=>c.abort();},[camera?.job_id]);
  useEffect(()=>{controller.current?.abort();setResult(null);setEvaluation(null);callback.current(null);setConfirmedState('현장 미확인');},[revision,dirty,time,at]);
  useEffect(()=>()=>controller.current?.abort(),[]);
  function changeCamera(id:string,patch:Partial<VenueCamera>){
    const c=current.current.config,old=c.cameras?.find(v=>v.camera_id===id);
    const sourceChanged=!!old&&((patch.job_id!==undefined&&patch.job_id!==old.job_id)||(patch.roi_name!==undefined&&patch.roi_name!==old.roi_name)||(patch.zone_id!==undefined&&patch.zone_id!==old.zone_id));
    onChange({...c,...(patch.active===false?{concentration_policy:{...c.concentration_policy!,center_observer_camera_id:c.concentration_policy?.center_observer_camera_id===id?null:c.concentration_policy?.center_observer_camera_id??null},target_policies:Object.fromEntries(Object.entries(c.target_policies??{}).map(([target,p])=>[target,{...p,center_observer_camera_id:p.center_observer_camera_id===id?null:p.center_observer_camera_id}]))}:{}),cameras:(c.cameras??[]).map(v=>v.camera_id===id?{...v,...patch,...(sourceChanged?{clock_verified:false,clock_evidence:'',full_zone_coverage:false}:{} )}:v),
      corridors:sourceChanged?(c.corridors??[]).map(g=>g.camera_id===id?{...g,boundary_verified:false,boundary_evidence:'',role:'approach',independent_counting_verified:false,independence_evidence:'',arrival_basis:'unknown',arrival_ratio:null,travel_seconds:null,arrival_evidence:''}:g):c.corridors});
  }
  function patchCamera(patch:Partial<VenueCamera>){if(camera)changeCamera(camera.camera_id,patch);}
  function patchGate(patch:Partial<Corridor>){const c=current.current.config;onChange({...c,corridors:(c.corridors??[]).map(g=>g.corridor_id===gateId?{...g,...patch}:g)});}
  function patchPolicy(patch:Partial<ConcentrationPolicy>){onChange({...config,concentration_policy:{...policy,...patch},target_policies:{...config.target_policies,...(config.center_zone_id?{[config.center_zone_id]:{...policy,...patch}}:{})}});}
  function addCamera(){
    let n=1;while(cameras.some(c=>c.camera_id===`CAM-${n}`))n++;
    const id=`CAM-${n}`;
    onChange({...config,cameras:[...cameras,{camera_id:id,name:`Camera ${n}`,registered_camera_id:null,point:[0,0],position_confirmed:false,bearing_deg:null,
      active:true,zone_id:config.zones[0]?.id??'',job_id:null,roi_name:'',coverage:[],full_zone_coverage:false,capture_start:null,clock_verified:false,clock_evidence:'',scenario_offset_s:0}]});
    setSelected(id);setMessage('카메라를 추가했습니다. 영상 ROI·지도 위치·관측 범위를 직접 지정한 뒤 지도 버전을 저장하세요.');
  }
  function addGate(){
    const origin=config.zones.find(z=>z.id!==config.center_zone_id);
    if(!camera||!origin||!config.center_zone_id){setMessage('카메라와 출발·중앙 구역을 먼저 선택하세요.');return;}
    const id=crypto.randomUUID();onChange({...config,corridors:[...corridors,{corridor_id:id,name:`중앙 통로 ${corridors.length+1}`,camera_id:camera.camera_id,from_zone_id:origin.id,to_zone_id:config.center_zone_id,
      boundary:[[0,0],[1,0]],center_side:[0,1],role:'approach',boundary_evidence:'',boundary_verified:false,counting_group:id,independent_counting_verified:false,independence_evidence:'',arrival_ratio:null,travel_seconds:null,arrival_basis:'unknown',arrival_evidence:''}]});setGateId(id);
  }
  async function estimate(){
    if(dirty||!revision)return;setBusy(true);setMessage('');const c=new AbortController();controller.current=c;
    const submitted={revision,time,at};
    try{
      const data=await estimateConcentration(revision,time,mode==='synchronized'&&commonAt?new Date(commonAt).toISOString():null,c.signal,venueId);
      if(c.signal.aborted||current.current.dirty||current.current.revision!==submitted.revision||Math.abs(current.current.time-submitted.time)>.001)return;
      setResult(data);callback.current(data);setConfirmedState('현장 미확인');
      setEvents(await getOperatorEvents(venueId));
    }catch(e){if(!c.signal.aborted)setMessage('통합 결과 조회 실패 — 저장 버전·촬영 시각·API 상태를 확인하세요.');}
    finally{setBusy(false);}
  }
  async function evaluate(){const savedRevision=revision;setBusy(true);try{const data=await evaluateConcentration(revision,venueId);if(!current.current.dirty&&current.current.revision===savedRevision)setEvaluation(data);}catch{setMessage('시간별 검증 실패 — 비교 가능한 중앙 관측과 충분한 과거·미래 이력이 필요합니다.');}finally{setBusy(false);}}
  async function record(){if(!result||!operator.trim())return;setBusy(true);try{
    await recordOperatorEvent({run_id:result.run_id,operator,zone_id:config.center_zone_id??null,verification,action,note},venueId);
    if(activeRun.current===result.run_id)setConfirmedState(verificationLabels[verification]);setEvents(await getOperatorEvents(venueId));setMessage('관리자 확인·대응 검토 기록을 저장했습니다. 외부 전송은 없습니다.');
  }catch{setMessage('확인 기록 저장 실패');}finally{setBusy(false);}}
  const centerName=config.zones.find(z=>z.id===config.center_zone_id)?.name??'중앙 구역 미설정';
  return <section className="concentration-panel" aria-label="다중 카메라 중앙 집중 관제">
    <header><div><span className="venue-eyebrow">MULTI-CAMERA / CENTRAL FLOW</span><h2>중앙 집중 추정 · {centerName}</h2></div><span className={`concentration-mode ${mode==='scenario'?'scenario':''}`}>{mode==='scenario'?'가상 시나리오 · 실제 동시 관측 아님':mode==='synchronized'?'촬영 시각 동기화 검증 모드':'독립 녹화 영상 · 합산 안 함'}</span></header>
    {message&&<p role="status" className="venue-notice">{message}</p>}
    <div className="concentration-toolbar"><button onClick={addCamera} disabled={cameras.length>=32||busy}>카메라 추가</button>{cameras.map(c=><button aria-pressed={selected===c.camera_id} key={c.camera_id} onClick={()=>{setSelected(c.camera_id);setGateId('');}}>{c.name}{!c.position_confirmed?' · 위치 미설정':''}</button>)}<span>카메라 {cameras.length}대 · 지도 매핑 v{revision}{dirty?' 초안 (미저장)':''}</span></div>
    <details className="concentration-settings"><summary>행사장·동기화·중앙 관측 정책 설정</summary><div className="concentration-form">
      <label>실제 행사장 주소<input aria-label="실제 행사장 주소" value={config.address??''} maxLength={500} onChange={e=>onChange({...config,address:e.target.value})}/></label>
      <label>중앙 집결 구역<select aria-label="중앙 집결 구역" value={config.center_zone_id??''} onChange={e=>{onChange({...config,target_policies:{...config.target_policies,...(config.center_zone_id?{[config.center_zone_id]:policy}:{})},center_zone_id:e.target.value||null,concentration_policy:config.target_policies?.[e.target.value]??{...defaultPolicy}});setGateId('');setMessage('예측 대상 구역을 변경했습니다. 구역별 통로와 정책은 보존됩니다.');}}><option value="">미설정</option>{config.zones.map(z=><option key={z.id} value={z.id}>{z.name}</option>)}</select></label>
      <label>영상 통합 모드<select aria-label="영상 통합 모드" value={mode} onChange={e=>onChange({...config,integration_mode:e.target.value as 'independent'|'scenario'|'synchronized'})}><option value="independent">독립 영상 (합산 금지)</option><option value="scenario">가상 시나리오 (가정으로만 통합)</option><option value="synchronized">검증된 촬영 시각 대응</option></select></label>
      <label>집계 이력 (초)<input aria-label="중앙 흐름 집계 이력" type="number" min={10} max={120} value={policy.window_seconds} onChange={e=>patchPolicy({window_seconds:Number(e.target.value)})}/></label>
      <label>중앙 관측 대표 카메라<select aria-label="중앙 관측 대표 카메라" value={policy.center_observer_camera_id??''} onChange={e=>patchPolicy({center_observer_camera_id:e.target.value||null})}><option value="">미관측</option>{cameras.filter(c=>c.zone_id===config.center_zone_id&&c.active!==false).map(c=><option key={c.camera_id} value={c.camera_id}>{c.name}</option>)}</select></label>
      <label><input type="checkbox" checked={policy.center_boundary_coverage_complete} onChange={e=>patchPolicy({center_boundary_coverage_complete:e.target.checked})}/> 중앙 출입 경계 전체 관측 확인</label>
      <label>전체 경계 관측 근거<input value={policy.coverage_evidence} onChange={e=>patchPolicy({coverage_evidence:e.target.value})}/></label>
      <label>주의 순유입 제안값 (회/분)<input aria-label="주의 순유입 기준" type="number" min="0.1" value={policy.attention_net_per_minute??''} onChange={e=>patchPolicy({attention_net_per_minute:e.target.value?Number(e.target.value):null})}/></label>
      <label>경계 수용 대비 비율 제안값<input type="number" min="0.01" step="0.05" value={policy.caution_capacity_ratio??''} onChange={e=>patchPolicy({caution_capacity_ratio:e.target.value?Number(e.target.value):null})}/></label>
      <label>긴급 확인 수용 대비 비율 제안값<input type="number" min="0.01" step="0.05" value={policy.urgent_capacity_ratio??''} onChange={e=>patchPolicy({urgent_capacity_ratio:e.target.value?Number(e.target.value):null})}/></label>
      <label><input type="checkbox" checked={policy.thresholds_validated} onChange={e=>patchPolicy({thresholds_validated:e.target.checked})}/> 관리자 운영 기준 검증 기록 있음</label><label>기준 검증 근거<input value={policy.threshold_evidence} onChange={e=>patchPolicy({threshold_evidence:e.target.value})}/></label>
    </div><p className="venue-muted">기본 안전 임계값은 제공하지 않습니다. 제안값과 현장 검증 기록을 구분합니다. 선택한 대상 구역별로 통로와 정책을 저장합니다.</p></details>
    {camera&&<><label><input type="checkbox" checked={camera.active!==false} onChange={e=>patchCamera({active:e.target.checked})}/> 카메라 활성</label></>}{camera&&<details className="concentration-settings" open><summary>{camera.name} · 영상·구역·경계선 매핑</summary><div className="concentration-form">
      <label>카메라 이름<input aria-label="배치 카메라 이름" value={camera.name} onChange={e=>patchCamera({name:e.target.value})}/></label><label>카메라 ID<input value={camera.camera_id} readOnly/></label>
      <label>담당 지도 구역<select aria-label="카메라 담당 구역" value={camera.zone_id} onChange={e=>patchCamera({zone_id:e.target.value,full_zone_coverage:false})}><option value="">구역 미지정</option>{config.zones.map(z=><option key={z.id} value={z.id}>{z.name}</option>)}</select></label>
      <label>완료 분석 작업 / 영상 소스<select aria-label="카메라 분석 작업" value={camera.job_id??''} onChange={e=>patchCamera({job_id:e.target.value||null,roi_name:'',clock_verified:false,capture_start:null,full_zone_coverage:false})}><option value="">영상 미연결</option>{jobs.filter(j=>j.status==='succeeded').map(j=><option key={j.id} value={j.id}>{j.analysis_mode==='test'?'테스트':'운영'} · 영상 {j.video_id.slice(0,8)} · {j.id.slice(0,8)}</option>)}</select></label>
      <label>분석 당시 영상 ROI<select aria-label="카메라 영상 ROI" value={camera.roi_name} onChange={e=>patchCamera({roi_name:e.target.value,full_zone_coverage:false})}><option value="">선택 필요</option>{summary?.roi_snapshot.zones.map(z=><option key={z.name} value={z.name}>{z.name}</option>)}</select></label>
      <label>기존 DB 카메라 ID (영상에 등록된 경우만)<input type="number" min="1" value={camera.registered_camera_id??''} onChange={e=>patchCamera({registered_camera_id:e.target.value?Number(e.target.value):null})}/></label>
      <label>지도 위치 X (0–1000)<input aria-label="카메라 지도 X" type="number" min="0" max="1000" value={camera.point[0]} onChange={e=>patchCamera({point:[Number(e.target.value),camera.point[1]],position_confirmed:true})}/></label>
      <label>지도 위치 Y (0–1000)<input aria-label="카메라 지도 Y" type="number" min="0" max="1000" value={camera.point[1]} onChange={e=>patchCamera({point:[camera.point[0],Number(e.target.value)],position_confirmed:true})}/></label>
      <button onClick={()=>{onPlaceCamera(camera.camera_id);setMessage('위쪽 행사장 지도에서 카메라 위치를 클릭하세요.');}}>지도에서 카메라 위치 지정</button>
      <label>지도상 촬영 방향 (도, 선택)<input type="number" min="0" max="359.99" value={camera.bearing_deg??''} onChange={e=>patchCamera({bearing_deg:e.target.value?Number(e.target.value):null})}/></label>
      <label>영상 첫 프레임 실제 촬영 시각 (시간대 포함)<input aria-label="카메라 촬영 시작 시각" placeholder="YYYY-MM-DDTHH:mm:ss+09:00" value={camera.capture_start??''} onChange={e=>patchCamera({capture_start:e.target.value||null,clock_verified:false})}/></label>
      <label><input aria-label="카메라 시각 검증" type="checkbox" checked={camera.clock_verified} onChange={e=>patchCamera({clock_verified:e.target.checked})}/> 촬영 시계·영상 시작 시각 확인</label>
      <label>시계 동기화 근거<input aria-label="시계 동기화 근거" value={camera.clock_evidence} onChange={e=>patchCamera({clock_evidence:e.target.value})}/></label>
      <label>시나리오 시작 오프셋 (초)<input type="number" min="0" value={camera.scenario_offset_s} onChange={e=>patchCamera({scenario_offset_s:Number(e.target.value)})}/></label>
      <label><input type="checkbox" checked={camera.full_zone_coverage} onChange={e=>patchCamera({full_zone_coverage:e.target.checked})}/> 담당 구역 전체를 해당 ROI로 관측함 (수동 확인)</label>
      <label>수동 관측 범위 지도 좌표 (JSON)<PolygonInput value={camera.coverage} onChange={p=>patchCamera({coverage:p})}/></label>
    </div><div className="concentration-toolbar"><a href={`/dashboard?venue=${encodeURIComponent(venueId)}&job=${encodeURIComponent(camera.job_id??'')}`} target="_blank" rel="noreferrer">연결 원본·YOLO·2D/3D 리플레이 열기</a><button disabled={!camera.job_id||camera.active===false} onClick={addGate}>중앙 방향 통로 추가</button><button onClick={()=>{onChange({...config,cameras:cameras.filter(c=>c.camera_id!==camera.camera_id),links:config.links?.map(link=>({...link,camera_ids:link.camera_ids.filter(id=>id!==camera.camera_id)})),corridors:corridors.filter(g=>g.camera_id!==camera.camera_id),target_policies:Object.fromEntries(Object.entries(config.target_policies??{}).map(([id,p])=>[id,{...p,center_observer_camera_id:p.center_observer_camera_id===camera.camera_id?null:p.center_observer_camera_id}])),concentration_policy:{...policy,center_observer_camera_id:policy.center_observer_camera_id===camera.camera_id?null:policy.center_observer_camera_id}});setSelected('');setGateId('');}}>카메라 배치 삭제</button></div>
      <div className="concentration-toolbar">{corridors.filter(g=>g.camera_id===selected).map(g=><button aria-pressed={gateId===g.corridor_id} key={g.corridor_id} onClick={()=>setGateId(g.corridor_id)}>{g.name}</button>)}</div>
      {gate&&<><div className="concentration-form"><label>통로 이름<input value={gate.name} onChange={e=>patchGate({name:e.target.value})}/></label><label>출발 구역<select value={gate.from_zone_id} onChange={e=>patchGate({from_zone_id:e.target.value})}>{config.zones.filter(z=>z.id!==config.center_zone_id).map(z=><option key={z.id} value={z.id}>{z.name}</option>)}</select></label><label>경계 의미<select aria-label="중앙 경계 의미" value={gate.role} onChange={e=>patchGate({role:e.target.value as Corridor['role'],boundary_verified:false})}><option value="approach">중앙 접근 경계 (도착 미확인)</option><option value="center_boundary">실제 중앙 출입 경계</option></select></label>
        <label><input aria-label="중앙 경계 검증" type="checkbox" checked={gate.boundary_verified} onChange={e=>patchGate({boundary_verified:e.target.checked})}/> 경계선과 중앙 측 방향 확인</label><label>실제 경계·방향 근거<input aria-label="경계 방향 근거" value={gate.boundary_evidence} onChange={e=>patchGate({boundary_evidence:e.target.value})}/></label>
        <label>집계 그룹 (겹치는 경계는 동일 그룹)<input aria-label="통로 집계 그룹" value={gate.counting_group} onChange={e=>patchGate({counting_group:e.target.value})}/></label><label><input type="checkbox" checked={gate.independent_counting_verified} onChange={e=>patchGate({independent_counting_verified:e.target.checked})}/> 다른 통로·카메라와 중복 집계 배제 확인</label><label>중복 배제 근거<input aria-label="중복 배제 근거" value={gate.independence_evidence} onChange={e=>patchGate({independence_evidence:e.target.value})}/></label>
        <label>도착률·이동 시간의 근거<select value={gate.arrival_basis} onChange={e=>patchGate({arrival_basis:e.target.value as Corridor['arrival_basis']})}><option value="unknown">미확보</option><option value="assumption">관리자 가정</option><option value="measured">측정 기록 있음</option></select></label><label>도착 비율 (0–1)<input type="number" min="0" max="1" step="0.05" value={gate.arrival_ratio??''} onChange={e=>patchGate({arrival_ratio:e.target.value?Number(e.target.value):null})}/></label><label>이동 시간 (초)<input type="number" min="0" value={gate.travel_seconds??''} onChange={e=>patchGate({travel_seconds:e.target.value?Number(e.target.value):null})}/></label><label>도착률·이동 시간 설명<input value={gate.arrival_evidence} onChange={e=>patchGate({arrival_evidence:e.target.value})}/></label>
      </div><p className="venue-muted">실제 중앙 출입 경계로 확인한 경우에만 그 경계의 통과를 중앙 유입·유출로 취급합니다. 접근 통로의 반대 이동을 중앙 유출로 간주하지 않습니다.</p>
      {summary&&<GateEditor key={gate.corridor_id} gate={gate} videoId={summary.video_id} resolution={summary.video_resolution} onChange={patchGate}/>}
      <button onClick={()=>{onChange({...config,corridors:corridors.filter(g=>g.corridor_id!==gate.corridor_id)});setGateId('');}}>통로 삭제</button></>}
    </details>}
    <div className="concentration-toolbar">{mode==='synchronized'&&<label>공통 촬영 기준 시각<input aria-label="공통 촬영 기준 시각" placeholder="YYYY-MM-DDTHH:mm:ss+09:00" readOnly={Number.isFinite(referenceMillis)} value={commonAt} onChange={e=>setAt(e.target.value)}/></label>}<button aria-label="중앙 집중 추정 조회" disabled={busy||dirty||!revision} onClick={()=>void estimate()}>{busy?'처리 중…':'저장된 매핑으로 현재 시점 조회'}</button><span>기준 영상 {time.toFixed(2)}초 · 조회 후 시간·설정이 바뀌면 결과 해제</span><button disabled={busy||dirty||!revision} onClick={()=>void evaluate()}>10·30·60초 과거 비교 검증</button></div>
    {dirty&&<p className="venue-notice">카메라·통로 설정을 아래 ‘새 설정 버전 저장’으로 먼저 저장하세요.</p>}
    {!result&&<p className="venue-muted">아직 이 시점의 통합 결과가 없습니다. 중앙 미관측은 0명이 아닙니다. 카메라가 동기화되지 않으면 개별 관측만 표시합니다.</p>}
    {result&&<div aria-label="중앙 집중 추정 결과" className="concentration-results">
      <div className="concentration-status"><strong>{result.state}</strong><span>{result.mode==='scenario'?'가상 시나리오 결과':'녹화 영상 분석 결과'} · {result.model_version}</span><span>{result.observed_at?timestamp(result.observed_at):`영상 기준 ${result.reference_offset_seconds.toFixed(2)}초 · 절대 시각 미확인`}</span></div>
      <p className="venue-muted">중앙 관측 인원 {number(result.observed_center_count)} · 전체 범위 확인 {result.center_total_coverage_verified?'등록됨':'미확인'} · 동시 증가 통로 {number(result.increasing_corridors)}개</p>
      {result.aggregation_reasons.length>0&&<p role="status" className="venue-notice">통합 제한: {result.aggregation_reasons.map(reasonLabel).join(' / ')}</p>}
      <div className="concentration-horizons">{result.estimates.map(e=><article key={e.horizon_seconds}><h3>{e.horizon_seconds}초 후 · {e.target_at?timestamp(e.target_at):`${e.target_offset_seconds.toFixed(2)}초`}</h3><strong>{e.expected_center_count!==null?`${number(e.expected_center_count)}명 (조건부 중앙 추정)`:`접근 압력 ${number(e.inward_pressure_per_minute)} 회/분`}</strong><p>예상 도착량 {number(e.expected_arrivals)} · 예상 유출량 {number(e.expected_departures)}</p><small>{e.confidence==='unavailable'?'추정 불가':'흐름 유지 가정 · 저신뢰·실제 다중 카메라 성능 미검증'}</small><p className="venue-muted">{e.reasons.map(reasonLabel).join(' / ')}</p></article>)}</div>
      <div className="concentration-table"><table><thead><tr>{['카메라 / 영상','기준 시각','관측 인원','이전 창 대비 인원','Optical Flow','저장 병목'].map(t=><th key={t}>{t}</th>)}</tr></thead><tbody>{result.cameras.map(c=><tr key={c.camera_id}><td>{c.name} / {c.job_id.slice(0,8)}{c.analysis_mode==='test'?' (테스트)':''}</td><td>{c.observed_at?timestamp(c.observed_at):`${c.offset_seconds??'미확인'}초`}</td><td>{number(c.count)}{c.reason&&<small>{reasonLabel(c.reason)}</small>}</td><td>{number(c.count_change)}</td><td>{c.metric?.low_confidence?'저신뢰':c.metric?.motion!=null?`${c.metric.motion.toFixed(4)} (픽셀 기반 이동량)`:'미분석'}<small>지도 방향으로 해석 안 함</small>{c.metric?.direction_counts&&<small>영상 방향 분포 (미보정): {Object.entries(c.metric.direction_counts).map(([d,n])=>`${d} ${n}`).join(' / ')}</small>}</td><td>{c.bottleneck?`${c.bottleneck.score} / ${c.bottleneck.severity} (사후 연구 지표)`:'미분석'}</td></tr>)}</tbody></table></div>
      <div className="concentration-table"><table><thead><tr>{['통로','중앙 방향 확인 통과','반대 방향 확인 통과','순통과','이전 창 대비 유입','품질·도착 근거'].map(t=><th key={t}>{t}</th>)}</tr></thead><tbody>{result.routes.map(r=><tr key={r.corridor_id}><td>{r.name}</td><td>{number(r.toward_center)}</td><td>{number(r.away_from_center)}</td><td>{number(r.net)}</td><td>{number(r.toward_change)}</td><td>{r.reason?reasonLabel(r.reason):'조건부 사용 / 관측 하한'}<small>{r.arrival_basis} · 도착 비율 {r.arrival_ratio??'미확보'} · 이동 {r.travel_seconds??'미확보'}초</small></td></tr>)}</tbody></table></div>
      <h3>관리자 사전 대응 검토</h3>{result.priority_corridors.map(r=><p key={r.corridor_id}>{r.name} · 중앙 방향 {number(r.toward_center)}회 — {r.recommendation}</p>)}<p className="venue-muted">고유 방문자 수·사고 발생 확률·필요 안전요원 수를 산출하지 않습니다. 검토 상태는 {result.threshold_basis==='administrator_validated'?'관리자 검증 기록':'검증 전 관리자 제안'} 기준입니다.</p>
      <div className="concentration-form"><label>확인 담당자<input aria-label="관제 확인 담당자" value={operator} onChange={e=>setOperator(e.target.value)} maxLength={100}/></label><label>현장 확인 상태<select aria-label="현장 확인 상태" value={verification} onChange={e=>setVerification(e.target.value)}>{Object.entries(verificationLabels).map(([id,label])=><option key={id} value={id}>{label}</option>)}</select></label><label>대응 검토<select value={action} onChange={e=>setAction(e.target.value)}><option value="review">상황 검토</option><option value="field_check">현장 점검</option><option value="staffing_review">인력 사전 배치 검토</option><option value="route_review">동선 조정 검토</option></select></label><label>관리자 대응 메모<textarea aria-label="관리자 대응 메모" rows={3} maxLength={4000} value={note} onChange={e=>setNote(e.target.value)}/></label></div><button disabled={busy||!operator.trim()} onClick={()=>void record()}>관리자 확인·메모 기록</button>
      <details><summary>신고 지원 정보 · 자동 전화·문자·신고 없음</summary><p className="venue-notice">실제 긴급 상황은 AI 결과나 화면 확인 절차를 기다리지 말고 관리자 판단으로 112 또는 119에 신고하세요.</p><textarea aria-label="신고 참고 정보" readOnly rows={13} value={incidentBrief(result,centerName,confirmedState,note)}/></details>
    </div>}
    {evaluation&&<section aria-label="중앙 추정 검증 지표"><h3>기록된 ROI 인원과 과거 시점 비교</h3><p className="venue-muted">{evaluation.evidence} · {evaluation.mode??'비교 자료 없음'} · 실제 동시 촬영·독립 현장 정답 검증과 다릅니다. 5분 예측은 지원하지 않습니다.</p><div className="concentration-toolbar">{evaluation.metrics.map(m=><span key={m.horizon_seconds}>{m.horizon_seconds}초: 표본 {m.samples} · MAE {number(m.mae)} · RMSE {number(m.rmse)} · {m.status}</span>)}</div></section>}
    {events.length>0&&<details><summary>최근 관리자 대응 이력</summary>{events.map(e=><p key={e.id}>{timestamp(e.created_at)} · {e.operator} · {verificationLabels[e.verification]} · {e.action} · {e.note} · 결과 {e.run_id.slice(0,8)}</p>)}</details>}
    <p className="venue-muted">팀원 모델 연동: 아티팩트 해시·입력 컬럼·대상·시간 범위·학습 단위·구역 ID·Python/패키지 버전·전처리·검증 지표·출력 예시를 받아 별도 어댑터에서 검증합니다. 현재 엔진은 팀원의 저장 모델을 실행하지 않습니다.</p>
  </section>;
}

function PolygonInput({value,onChange}:{value:Point[];onChange:(v:Point[])=>void}){
  const [text,setText]=useState(JSON.stringify(value)),[error,setError]=useState('');
  useEffect(()=>setText(JSON.stringify(value)),[value]);
  return <><textarea aria-label="카메라 관측 범위 좌표" rows={2} value={text} onChange={e=>setText(e.target.value)} onBlur={()=>{try{const v=JSON.parse(text);if(!Array.isArray(v)||v.length>100||v.some(p=>!Array.isArray(p)||p.length!==2||p.some((n:unknown)=>typeof n!=='number'||!Number.isFinite(n)||n<0||n>1000)))throw new Error();onChange(v);setError('');}catch{setError('0–1000 좌표 쌍의 배열을 입력하세요.');}}}/>{error&&<small role="alert">{error}</small>}</>;
}

function GateEditor({gate,videoId,resolution,onChange}:{gate:Corridor;videoId:string;resolution:Point;onChange:(p:Partial<Corridor>)=>void}){
  const [step,setStep]=useState<number|null>(null);
  return <div className="gate-editor"><div className="concentration-toolbar"><button onClick={()=>setStep(step===null?0:null)}>{step===null?'영상에서 경계선·중앙 측 지정':'경계 편집 취소'}</button><span>{step===0?'경계 시작점을 클릭하세요.':step===1?'경계 끝점을 클릭하세요.':step===2?'중앙으로 이어지는 측의 한 점을 클릭하세요.':'영상 픽셀 좌표 · 중앙 측은 지도에서 확인한 통로 기준'}</span></div>
    <div className="gate-viewport" style={{aspectRatio:`${resolution[0]}/${resolution[1]}`}}><video src={api.getUri({url:`/api/videos/${encodeURIComponent(videoId)}/stream`})} controls playsInline preload="metadata" aria-label="카메라 경계 설정 영상"/><svg aria-label="중앙 방향 경계 편집" viewBox={`0 0 ${resolution[0]} ${resolution[1]}`} style={{pointerEvents:step===null?'none':'auto'}} onPointerDown={e=>{if(step===null)return;const p=screenToVideo(e.clientX,e.clientY,e.currentTarget.getBoundingClientRect(),resolution);if(!p)return;if(step===0)onChange({boundary:[p,gate.boundary[1]],boundary_verified:false});if(step===1)onChange({boundary:[gate.boundary[0],p],boundary_verified:false});if(step===2)onChange({center_side:p,boundary_verified:false});setStep(step===2?null:step+1);}}><line x1={gate.boundary[0][0]} y1={gate.boundary[0][1]} x2={gate.boundary[1][0]} y2={gate.boundary[1][1]} stroke="#f0c864" strokeWidth="3" vectorEffect="non-scaling-stroke"/><circle cx={gate.center_side[0]} cy={gate.center_side[1]} r="10" fill="#6dc9d9"/><text x={gate.center_side[0]+15} y={gate.center_side[1]} fontSize="25" fill="#fff">중앙 측 (수동 설정)</text></svg></div>
    <div className="concentration-form">{[0,1,2].map(i=><label key={i}>{['경계 시작','경계 끝','중앙 측 기준점'][i]}{(i===2?gate.center_side:gate.boundary[i]).map((n,j)=><input aria-label={`${['경계 시작','경계 끝','중앙 측 기준점'][i]} ${j===0?'X':'Y'}`} key={j} type="number" min="0" max={resolution[j]} value={n} onChange={e=>{const p=[...(i===2?gate.center_side:gate.boundary[i])] as Point;p[j]=Number(e.target.value);if(i===2)onChange({center_side:p,boundary_verified:false});else{const b=[...gate.boundary] as [Point,Point];b[i]=p;onChange({boundary:b,boundary_verified:false});}}}/>)}</label>)}</div>
  </div>;
}
