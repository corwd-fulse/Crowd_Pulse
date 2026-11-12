import {useEffect, useRef, useState} from 'react';
import axios from 'axios';
import DetectionOverlay from './DetectionOverlay';
import VideoZoneCounts from './VideoZoneCounts';
import {api, getVideoRoi, registerVideoRoi} from '../../services/api';
import type {RoiSnapshot} from '../../services/api';
import {screenToVideo, validateZones} from '../../services/roiGeometry';
import type {Point, Zone} from '../../services/roiGeometry';

export interface RoiReadiness {videoId: string; production: boolean; test: boolean}
const colors=['#23cfff','#ffc653','#23e3bb','#ff8db3','#ab9bff'];
const button='rounded-sm border border-line px-3 py-2 text-xs text-ink-soft disabled:cursor-not-allowed disabled:opacity-40';

export default function VideoRoiEditor({videoId,onReadiness,detectionJobId=null,analysisMessage='아직 분석하지 않은 영상입니다'}: {videoId:string;onReadiness:(state:RoiReadiness)=>void;detectionJobId?:string|null;analysisMessage?:string}) {
  const video=useRef<HTMLVideoElement>(null);
  const [showOriginal,setShowOriginal]=useState(true);
  const [showRoi,setShowRoi]=useState(true);
  const [showDetections,setShowDetections]=useState(true);
  const [source,setSource]=useState<RoiSnapshot|null|undefined>(undefined);
  const [zones,setZones]=useState<Zone[]>([]);
  const [resolution,setResolution]=useState<Point>([0,0]);
  const [selected,setSelected]=useState<number|null>(null);
  const [points,setPoints]=useState<Point[]>([]);
  const [name,setName]=useState('');
  const [dirty,setDirty]=useState(false);
  const [preview,setPreview]=useState(false);
  const [saving,setSaving]=useState(false);
  const [message,setMessage]=useState('');
  const [error,setError]=useState('');
  const [drawError,setDrawError]=useState('');
  const [surfaceWidth,setSurfaceWidth]=useState(0);
  const drawAlert=useRef<HTMLParagraphElement>(null);
  const [reload,setReload]=useState(0);
  const svg=useRef<SVGSVGElement>(null);
  const drag=useRef<{zone:number;vertex:number;pointer:number}|null>(null);
  const saveLock=useRef(false);
  const resolutionWidth=resolution[0];
  const roiUnavailable=source===undefined;
  useEffect(()=>{
    const element=svg.current;
    if(!element || typeof ResizeObserver==='undefined') return;
    const observer=new ResizeObserver(entries=>setSurfaceWidth(entries[0].contentRect.width));
    observer.observe(element);
    return ()=>observer.disconnect();
  },[resolutionWidth,roiUnavailable]);
  useEffect(()=>{
    if(drawError) {
      drawAlert.current?.scrollIntoView?.({block:'nearest',behavior:'smooth'});
      drawAlert.current?.focus({preventScroll:true});
    }
  },[drawError]);
  useEffect(()=>{
    const controller=new AbortController();
    getVideoRoi(videoId,controller.signal).then(result=>{
      if(controller.signal.aborted) return;
      setSource(result?.roi_snapshot??null);setZones(structuredClone(result?.roi_snapshot.zones??[]));
      setSelected(result?.roi_snapshot.zones.length?0:null);setDirty(false);setPoints([]);setError('');setDrawError('');
    }).catch(()=>{if(!controller.signal.aborted) setError('ROI 조회에 실패했습니다. 다시 조회하세요.');});
    return ()=>controller.abort();
  },[videoId,reload]);
  const errors=validateZones(zones,resolution);
  const mismatch=!!source&&(source.video_resolution[0]!==resolution[0]||source.video_resolution[1]!==resolution[1]);
  const ready=!!source&&!dirty&&!points.length&&!errors.length&&!mismatch&&!saving;
  useEffect(()=>onReadiness({videoId,production:ready&&source?.status==='approved'&&source.approved_for_analysis===true,test:ready}),[videoId,ready,source,onReadiness]);
  const candidate:RoiSnapshot={...(source??{}),status:'candidate',approved_for_analysis:false,video_resolution:resolution,zones:zones.map(z=>({...z,name:z.name.trim()}))};
  function changed(next:Zone[]) {setZones(next);setDirty(true);setMessage('');setError('');setDrawError('');}
  function finish() {
    const next=[...zones,{name:name.trim(),polygon:points}];
    const problems=validateZones(next,resolution);
    if(problems.length) {setDrawError(problems.join(' '));return;}
    changed(next);setSelected(next.length-1);setPoints([]);setName('');
  }
  async function save() {
    if(saveLock.current||source!==null||errors.length||points.length||!preview) return;
    saveLock.current=true;setSaving(true);setError('');setMessage('');
    try {
      const result=await registerVideoRoi(videoId,candidate);
      setSource(result.roi_snapshot);setDirty(false);setMessage('후보 ROI를 저장했습니다. 운영 승인은 별도 절차가 필요합니다.');
    } catch(e) {
      setError(axios.isAxiosError(e)&&e.response?.status===412?'이미 ROI가 등록되었습니다. 덮어쓰지 않았습니다. 다시 조회하세요.':'ROI 저장에 실패했습니다. 연결 상태를 확인하세요.');
    } finally {saveLock.current=false;setSaving(false);}
  }
  function exportDraft() {
    const url=URL.createObjectURL(new Blob([JSON.stringify(candidate,null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download=`roi-candidate-${videoId}.json`;a.click();URL.revokeObjectURL(url);
  }
  const disabled=source===undefined||saving||!resolution[0];
  return <section aria-label="영상 ROI 설정" className="overflow-hidden rounded-lg border border-line bg-surface">
    <div className="space-y-2 border-b border-line p-5"><h2 className="text-sm font-semibold text-ink">영상 구역(ROI) 설정</h2>
      <p className="text-xs text-muted">원본 해상도 {resolution[0]||'—'} × {resolution[1]||'—'} · {roiUnavailable?(error?'ROI 조회 실패 — 편집 불가':'ROI 확인 중'):source===null?'서버 ROI 미등록 — 미저장 초안':dirty?'미승인 후보 초안':source?.approved_for_analysis&&source.status==='approved'?'승인 ROI':'서버에 저장된 미승인 후보 ROI'}{dirty?' · 저장되지 않은 초안':''}</p>
      <p className="text-xs text-muted">편집 모드에서 클릭·터치로 꼭짓점을 추가하고 완료 버튼 또는 더블클릭으로 확정하세요. 구역을 선택하면 꼭짓점을 드래그할 수 있습니다. 미리보기 모드에서는 영상을 재생해 구역을 확인하세요.</p>
      {source&&<p className="text-xs text-warn">기존 ROI는 보존합니다. 수정·삭제한 초안은 JSON으로 내보낼 수 있습니다. 새 버전 저장 및 승인 API는 아직 없습니다.</p>}
    </div>
    <div className="flex flex-wrap items-center gap-4 border-b border-line px-5 py-3 text-xs text-ink-soft">
      <label><input type="checkbox" checked={showOriginal} onChange={e=>setShowOriginal(e.target.checked)}/> 원본 영상</label>
      <label><input type="checkbox" checked={showDetections} onChange={e=>setShowDetections(e.target.checked)}/> YOLO 탐지</label>
      <label><input type="checkbox" checked={showRoi} disabled={!!points.length} onChange={e=>{setShowRoi(e.target.checked);if(!e.target.checked)setPreview(true);}}/> ROI 구역</label>
      <button type="button" className={button} onClick={()=>{const v=video.current;if(v){if(v.paused)void v.play().catch(()=>setError('영상을 재생할 수 없습니다.'));else v.pause();}}}>재생 / 일시정지</button>
    </div>
    <div className="bg-monitor p-3">
      <div className="relative mx-auto w-full max-w-5xl" style={resolution[0]?{aspectRatio:`${resolution[0]} / ${resolution[1]}`}:{}}>
        <video ref={video} style={{opacity:showOriginal?1:0}} aria-label="ROI 원본 영상" className="block h-full w-full" src={api.getUri({url:`/api/videos/${encodeURIComponent(videoId)}/stream`})} controls playsInline preload="metadata" onLoadedMetadata={e=>setResolution([e.currentTarget.videoWidth,e.currentTarget.videoHeight])} onError={()=>setError('원본 영상을 불러오지 못했습니다. 영상과 서버 연결을 확인하세요.')}/>
        {!!resolution[0]&&source!==undefined&&<svg ref={svg} aria-label="ROI 그리기 영역" role="img" style={{opacity:showRoi?1:0}} viewBox={`0 0 ${resolution[0]} ${resolution[1]}`} className={`absolute inset-0 h-full w-full ${preview||!showRoi?'pointer-events-none':'touch-none'}`}
          onClick={e=>{if(disabled||preview||selected!==null||e.detail>1)return;const p=screenToVideo(e.clientX,e.clientY,e.currentTarget.getBoundingClientRect(),resolution);if(p){setPoints(previous=>[...previous,p]);setMessage('');setDrawError('');}}}
          onDoubleClick={()=>{if(!preview&&selected===null)finish();}}
          onPointerMove={e=>{const d=drag.current;if(!d||d.pointer!==e.pointerId||preview||saving)return;const p=screenToVideo(e.clientX,e.clientY,e.currentTarget.getBoundingClientRect(),resolution);if(p)changed(zones.map((z,i)=>i===d.zone?{...z,polygon:z.polygon.map((v,k)=>k===d.vertex?p:v)}:z));}}
          onPointerUp={e=>{if(drag.current?.pointer===e.pointerId){drag.current=null;e.currentTarget.releasePointerCapture(e.pointerId);}}}
          onPointerCancel={()=>{drag.current=null;}}>
          {zones.map((z,i)=><g key={i}><polygon points={z.polygon.map(p=>p.join(',')).join(' ')} fill={colors[i%colors.length]} fillOpacity={.16} stroke={colors[i%colors.length]} strokeWidth={selected===i?3:2} vectorEffect="non-scaling-stroke" onClick={e=>{if(selected===null)return;e.stopPropagation();if(!disabled&&!preview&&!points.length){setSelected(i);setDrawError('');}}}/><text x={z.polygon[0]?.[0]} y={(z.polygon[0]?.[1]??0)+resolution[0]*.016} fill={colors[i%colors.length]} fontSize={resolution[0]*.017} className="pointer-events-none">{z.name}</text>
            {selected===i&&!preview&&z.polygon.map((p,k)=><circle key={k} aria-label={`${z.name} 꼭짓점 ${k+1}`} cx={p[0]} cy={p[1]} r={22*resolution[0]/(surfaceWidth||resolution[0])} fill="transparent" className="cursor-move" onPointerDown={e=>{if(disabled)return;e.stopPropagation();const position=svg.current&&screenToVideo(e.clientX,e.clientY,svg.current.getBoundingClientRect(),resolution);const vertex=position?z.polygon.reduce((best,p,index)=>Math.hypot(p[0]-position[0],p[1]-position[1])<Math.hypot(z.polygon[best][0]-position[0],z.polygon[best][1]-position[1])?index:best,k):k;drag.current={zone:i,vertex,pointer:e.pointerId};svg.current?.setPointerCapture(e.pointerId);}}><title>드래그로 꼭짓점 이동</title></circle>)}
            {selected===i&&z.polygon.map((p,k)=><circle key={k} cx={p[0]} cy={p[1]} r={5*resolution[0]/(surfaceWidth||resolution[0])} fill={colors[i%colors.length]} className="pointer-events-none"/>)}</g>)}
          <polyline points={points.map(p=>p.join(',')).join(' ')} fill="none" stroke="#ffffff" strokeWidth={2} vectorEffect="non-scaling-stroke" className="pointer-events-none"/>
          {points.map((p,k)=><circle key={k} cx={p[0]} cy={p[1]} r={5*resolution[0]/(surfaceWidth||resolution[0])} fill="white" className="pointer-events-none"/>)}
        </svg>}
        {!!resolution[0]&&<DetectionOverlay jobId={detectionJobId} videoId={videoId} video={video} resolution={resolution} show={showDetections} emptyMessage={analysisMessage}/>}
      </div>
    </div>
    {!!resolution[0]&&<VideoZoneCounts key={detectionJobId??'none'} jobId={detectionJobId} videoId={videoId} video={video}/>}
    <div className="space-y-4 p-5">
      <div className="flex flex-wrap gap-2"><button type="button" className={button} disabled={disabled||!!points.length} onClick={()=>setPreview(v=>!v)}>{preview?'구역 편집':'저장 전 미리보기'}</button><button type="button" className={button} disabled={disabled||preview||!showRoi||!!points.length} onClick={()=>{setSelected(null);setName(`구역 ${zones.length+1}`);setDrawError('');}}>새 구역 그리기</button><button type="button" className={button} disabled={saving} onClick={()=>{if(!dirty&&!points.length||window.confirm('저장하지 않은 초안을 버리고 다시 조회할까요?')){setSource(undefined);setZones([]);setSelected(null);setPoints([]);setMessage('');setError('');setDrawError('');setReload(v=>v+1);}}}>ROI 다시 조회</button></div>
      {source===undefined&&!error&&<p role="status" className="text-xs text-muted">ROI를 불러오는 중…</p>}
      {!preview&&selected===null&&<div className="flex flex-wrap items-end gap-2"><label className="text-xs text-ink-soft">새 구역 이름<input aria-label="새 구역 이름" maxLength={100} value={name} disabled={disabled} onChange={e=>{setName(e.target.value);setDrawError('');}} className="mt-1 block w-44 rounded-sm border border-line bg-surface p-2"/></label><button type="button" className={button} disabled={disabled||points.length<3} onClick={finish}>폴리곤 완료 ({points.length}점)</button><button type="button" className={button} disabled={!points.length||saving} onClick={()=>{setPoints(p=>p.slice(0,-1));setDrawError('');}}>마지막 점 취소</button><button type="button" className={button} disabled={!points.length||saving} onClick={()=>{setPoints([]);setDrawError('');}}>그리기 취소</button></div>}
      {drawError&&<p ref={drawAlert} tabIndex={-1} role="alert" className="rounded-sm border border-rose-500/40 p-3 text-xs text-danger">{drawError}</p>}
      <div className="space-y-2">{zones.map((z,i)=><div key={i} className="flex flex-wrap items-center gap-2"><button type="button" className={button} disabled={saving||!!points.length} onClick={()=>{setSelected(i);setPreview(false);}} style={{borderColor:colors[i%colors.length]}}>구역 {i+1} 선택</button><input aria-label={`구역 ${i+1} 이름`} maxLength={100} value={z.name} disabled={disabled||preview} onChange={e=>changed(zones.map((v,k)=>k===i?{...v,name:e.target.value}:v))} className="min-w-0 flex-1 rounded-sm border border-line bg-surface p-2 text-xs text-ink"/><button type="button" className={button} disabled={disabled||preview} onClick={()=>{changed(zones.filter((_,k)=>k!==i));setSelected(null);}}>구역 {i+1} 삭제</button></div>)}</div>
      {mismatch&&<p role="alert" className="text-xs text-danger">기존 ROI 해상도와 원본 영상이 다릅니다. 분석 요청을 차단했습니다.</p>}
      {!!resolution[0]&&errors.length>0&&<ul aria-label="ROI 검증 결과" className="list-inside list-disc text-xs text-warn">{errors.map(e=><li key={e}>{e}</li>)}</ul>}
      {preview&&!errors.length&&<p className="text-xs text-ok">좌표 범위·이름·면적·자기 교차·구역 간 경계 검사를 통과했습니다. 실제 보행 영역과 영상 전체에서의 적합성은 직접 확인하세요.</p>}
      <details className="text-xs text-muted"><summary>원본 좌표 JSON 미리보기</summary><pre className="mt-2 max-h-60 overflow-auto">{JSON.stringify(candidate,null,2)}</pre></details>
      <div className="flex flex-wrap gap-2"><button type="button" className={button} disabled={disabled||source!==null||!!errors.length||!!points.length||!preview} onClick={()=>void save()}>{saving?'저장 중…':'후보 ROI 최초 저장'}</button><button type="button" className={button} disabled={disabled||!!errors.length||!!points.length} onClick={exportDraft}>후보 초안 JSON 내보내기</button></div>
      {message&&<p role="status" className="text-xs text-ok">{message}</p>}{error&&<p role="alert" className="text-xs text-danger">{error}</p>}
    </div>
  </section>;
}
