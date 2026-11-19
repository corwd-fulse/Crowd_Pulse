import {useMediaWindow} from '../../hooks/useMediaWindow';
import {lazy,Suspense,useEffect,useMemo,useRef,useState} from 'react';
import type {RefObject,ReactNode} from 'react';
import {getReplay,getVisionZoneMetrics,getVisionBottleneckScores} from '../../services/api';
import type {ReplayResult,VisionResultSummary,VisionZoneMetric,VisionBottleneckScore} from '../../services/api';
import {replayFrameAt,canvasPoint} from '../../services/replayTiming';
import {selectReplayScene,threePreference} from '../../services/replayScene';
import Replay3DBoundary from './Replay3DBoundary';

const Replay3D=lazy(()=>import('./Replay3D'));

const palette=['#23cfff','#ffc653','#23e3bb','#ff8db3','#ab9bff'];
const diagnosticLabels:Record<string,string>={id_gap_or_duplicate:'ID 단절',suspected_id_switch:'ID 교환 의심',ambiguous_overlap:'구역 겹침',missing_track_id:'ID 없음',unobserved_origin:'유입 경로 미확인',unobserved_destination:'유출 경로 미확인'};
const metric=(n:number|null|undefined)=>n==null?'미분석':n.toLocaleString('ko-KR');

export default function ZoneReplay({jobId,videoId,video,resolution,summary,videoPane}:{jobId:string;videoId:string;video:RefObject<HTMLVideoElement|null>;resolution:[number,number];summary:VisionResultSummary|null;videoPane?:ReactNode}) {
  const canvas=useRef<HTMLCanvasElement>(null);
  const [time,setTime]=useState(0);
  const page=useMediaWindow<ReplayResult>(jobId,videoId,time,getReplay);
  const [playError,setError]=useState('');
  const data=page.data,error=playError||(page.error?'Replay window unavailable':'');
  const [rate,setRate]=useState(1),[route,setRoute]=useState('');
  const [canvasWidth,setCanvasWidth]=useState(960);
  const [viewMode,setViewMode]=useState<'2d'|'3d'>('2d');
  const [viewNotice,setViewNotice]=useState(''),[zoneFilter,setZoneFilter]=useState('');
  const fallback3D=(reason:string)=>{setViewMode('2d');setViewNotice(reason);};
  function request3D(){
    try {
      const probe=document.createElement('canvas');
      const context=probe.getContext('webgl2');
      if(!context){fallback3D('이 브라우저에서 WebGL2를 사용할 수 없어 2D를 유지합니다.');return;}
      context.getExtension('WEBGL_lose_context')?.loseContext();
      setViewNotice('');setViewMode('3d');
    }catch{fallback3D('3D 그래픽을 사용할 수 없어 2D를 유지합니다.');}
  }
  const metricStart=Math.floor(time/10)*10000;
  const [metrics,setMetrics]=useState<VisionZoneMetric[]>([]),[scores,setScores]=useState<VisionBottleneckScore[]>([]);
  useEffect(()=>{
    const controller=new AbortController();
    async function metricsPages(){
      const rows:VisionZoneMetric[]=[],risk:VisionBottleneckScore[]=[];
      for(let offset=0;;offset+=1000){const p=await getVisionZoneMetrics(jobId,offset,controller.signal,{start_ms:metricStart,end_ms:metricStart+10000});rows.push(...p.items);if(p.items.length<1000)break;}
      for(let offset=0;;offset+=1000){const p=await getVisionBottleneckScores(jobId,offset,controller.signal,{start_ms:metricStart,end_ms:metricStart+10000});risk.push(...p.items);if(p.items.length<1000)break;}
      if([...rows,...risk].some(r=>r.analysis_job_id!==jobId))throw new Error('Mismatched replay metrics');
      if(!controller.signal.aborted){setMetrics(rows);setScores(risk);}
    }
    void metricsPages().catch(()=>{if(!controller.signal.aborted){setMetrics([]);setScores([]);}});
    return ()=>controller.abort();
  },[jobId,videoId,metricStart]);
  useEffect(()=>{
    const element=video.current;if(!element)return;
    let stopped=false,id=0;
    const update=(event?:Event)=>{
      // During playback use decoded video frames, not an extra partial-time tick.
      if(event?.type!=='timeupdate'||element.paused||!element.requestVideoFrameCallback)setTime(element.currentTime);
      setRate(element.playbackRate);
    };
    const tick:VideoFrameRequestCallback=(_,metadata)=>{if(stopped)return;setTime(metadata.mediaTime);id=element.requestVideoFrameCallback(tick);};
    if(element.requestVideoFrameCallback)id=element.requestVideoFrameCallback(tick);
    for(const name of ['seeked','timeupdate','loadedmetadata','ratechange','pause','play'])element.addEventListener(name,update);
    update();return()=>{stopped=true;if(id)element.cancelVideoFrameCallback(id);for(const name of ['seeked','timeupdate','loadedmetadata','ratechange','pause','play'])element.removeEventListener(name,update);};
  },[video]);
  const mismatch=!!data&&data.video_resolution.some((n,i)=>n!==resolution[i]);
  useEffect(()=>{
    const element=canvas.current;if(!element||typeof ResizeObserver==='undefined')return;
    const observer=new ResizeObserver(entries=>{if(entries[0].contentRect.width>0)setCanvasWidth(entries[0].contentRect.width);});
    observer.observe(element);return()=>observer.disconnect();
  },[data,mismatch,viewMode]);
  const frame=data&&!mismatch?replayFrameAt(data,time):undefined;
  const selectedRoute=route?JSON.parse(route) as [string,string]:null;
  const scene=useMemo(()=>data?selectReplayScene(data,frame,time,{route:route?JSON.parse(route):null,zone:zoneFilter?JSON.parse(zoneFilter):undefined}):null,[data,frame,time,route,zoneFilter]);
  // Allocate once per replay, including all filters and observed headcounts.
  const capacity=useMemo(()=>data?.frames.reduce((max,f)=>Math.max(max,f.points.length),1)??1,[data]);
  const preference=threePreference({width:typeof window==='undefined'?1024:window.innerWidth,cores:navigator.hardwareConcurrency,memory:(navigator as Navigator&{deviceMemory?:number}).deviceMemory});
  useEffect(()=>{
    const element=canvas.current;if(!element||!data||mismatch||!scene||viewMode!=='2d')return;
    const ctx=element.getContext('2d');if(!ctx)return;
    const width=Math.round(canvasWidth),height=Math.round(width*resolution[1]/resolution[0]);element.width=width;element.height=height;
    const point=(x:number,y:number)=>canvasPoint(x,y,resolution,width,height);
    ctx.fillStyle='#071a30';ctx.fillRect(0,0,width,height);
    for(const [i,z] of data.roi_snapshot.zones.entries()){
      if(z.visible===false)continue;
      ctx.beginPath();z.polygon.forEach(([x,y],j)=>{const [cx,cy]=point(x,y);if(j===0)ctx.moveTo(cx,cy);else ctx.lineTo(cx,cy);});ctx.closePath();
      ctx.fillStyle=(z.color??palette[i%palette.length])+'22';ctx.fill();ctx.strokeStyle=z.color??palette[i%palette.length];ctx.lineWidth=2;ctx.stroke();
      const [x,y]=point(...z.polygon[0]);ctx.font='14px sans-serif';ctx.fillStyle=z.color??palette[i%palette.length];ctx.fillText(z.name,x+5,y+18);
    }
    for(const e of scene.events){
      const [x,y]=point(...e.from_position),[a,b]=point(...e.to_position);const angle=Math.atan2(b-y,a-x);
      ctx.strokeStyle=e.from_zone.localeCompare(e.to_zone)>0?'#ff8db3':'#ffc653';ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(a,b);ctx.stroke();
      ctx.beginPath();ctx.moveTo(a,b);ctx.lineTo(a-10*Math.cos(angle-.4),b-10*Math.sin(angle-.4));ctx.lineTo(a-10*Math.cos(angle+.4),b-10*Math.sin(angle+.4));ctx.fill();
    }
    for(const p of scene.actors){
      const [x,y]=point(p.x,p.y);const uncertain=p.uncertain;
      ctx.fillStyle=uncertain?'#ff8db3':'#23e3bb';ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill();ctx.font='12px sans-serif';ctx.fillText(String(p.trackId),x+7,y-5);
    }
    if(!frame){ctx.fillStyle='#ffc653';ctx.font='18px sans-serif';ctx.fillText('이 시각의 관측 기록 없음',20,40);}
  },[data,frame,time,route,mismatch,resolution,canvasWidth,scene,viewMode]);
  const duration=data?data.frame_count/data.fps:video.current?.duration;
  const currentMetrics=metrics.filter(r=>Math.floor(r.video_offset_ms/1000)===Math.floor(time));
  const currentScores=scores.filter(r=>{
    if(!frame?.observed||r.window_start_ms!==Math.floor(time/10)*10000)return false;
    const status=summary?.bottleneck_status?.find(s=>s.zone===r.region_key&&s.window_start*1000===r.window_start_ms);
    return status?.status!=='unanalyzed'&&!Object.values(status?.missing_metrics??{}).some(v=>v.length)&&!Object.values(status?.previous_missing_metrics??{}).some(v=>v.length);
  });
  const recentIssues=data?.diagnostics.filter(d=>d.frame<=Math.floor(time*(data?.fps??25))&&d.frame>=Math.floor(time*(data?.fps??25))-25)??[];
  return <section aria-label="실제 추적 2D 리플레이" data-view-mode={viewMode} data-scene-frame={scene?.frame??''} data-visible-track-ids={JSON.stringify(scene?.actors.map(a=>a.trackId)??[])} data-active-event-ids={JSON.stringify(scene?.events.map(e=>e.event_id)??[])} className="min-w-0 rounded-lg border border-line bg-surface p-3 text-xs text-ink-soft">
    <h3 className="font-semibold text-ink">{viewMode==='2d'?'2D 관측 리플레이':'3D 디지털 트윈 · 관측 리플레이'}</h3>
    <p className="my-2 text-muted">실제 추적 기록 재생 · 미래 예측 아님 · 점은 bbox 하단 중앙 · 화살표는 확인된 전환 방향이며 보간 경로가 아닙니다.</p>
    {!error&&data&&!mismatch&&<>
      <div className="mb-2 flex flex-wrap gap-2"><button type="button" aria-pressed={viewMode==='2d'} onClick={()=>setViewMode('2d')} className="rounded-sm border border-line px-3 py-1">2D 보기</button><button type="button" aria-pressed={viewMode==='3d'} onClick={request3D} className="rounded-sm border border-brand px-3 py-1 text-brand">3D 보기</button><label>구역 <select aria-label="리플레이 구역 필터" value={zoneFilter} onChange={e=>setZoneFilter(e.target.value)} className="bg-surface"><option value="">전체 구역</option>{data.roi_snapshot.zones.map(z=><option key={z.name} value={JSON.stringify(z.name)}>{z.name}</option>)}<option value="null">구역 밖</option></select></label></div>
      {preference&&<p className="mb-2 text-muted">{preference} 3D 선택 시 저사양 렌더링으로 시작합니다.</p>}
      {viewNotice&&<p role="status" className="mb-2 text-warn">{viewNotice}</p>}
      <p className="mb-2 text-muted">현재 표시: 유효 ID {scene?.actors.length??0}명{scene?.unidentified?` · ID 미확인 ${scene.unidentified}개 관측은 캐릭터로 표시하지 않음`:''}</p>
    </>}
      <div className={`grid min-w-0 gap-3 ${videoPane?'xl:grid-cols-2':''}`}>
        {videoPane}
        <div data-replay-viewport className="relative min-w-0 overflow-hidden rounded-sm border border-line bg-[#0b1422]" style={{aspectRatio:resolution[0]?`${resolution[0]} / ${resolution[1]}`:'16 / 9'}}>
          {error?<p role="alert" className="p-4 text-warn">{error}</p>:!data?<p role="status" className="p-4">리플레이 기록 로딩 중…</p>:mismatch?<p role="alert" className="p-4">영상과 리플레이 해상도가 달라 표시하지 않습니다.</p>:viewMode==='2d'?<canvas ref={canvas} aria-label="구역별 추적 위치 캔버스" data-replay-frame={frame?.frame??''} data-route-filter={route} data-2d-points={JSON.stringify(scene?.actors.map(a=>({id:a.trackId,x:a.x,y:a.y}))??[])} className="block h-full w-full"/>:
            scene&&<Replay3DBoundary onFallback={fallback3D}><Suspense fallback={<p role="status" className="py-10">3D 화면 로딩 중… 2D 보기는 계속 사용할 수 있습니다.</p>}><Replay3D scene={scene} roi={data.roi_snapshot} resolution={data.video_resolution} time={time} onFallback={fallback3D} capacity={capacity} fitViewport/></Suspense></Replay3DBoundary>}
        </div>
      </div>
    {!error&&data&&!mismatch&&<>
      <div className="my-2 flex flex-wrap items-center gap-2"><button type="button" className="rounded-sm border border-line px-2 py-1" onClick={()=>{const v=video.current;if(v){if(v.paused)void v.play().catch(()=>setError('원본 영상 재생 실패'));else v.pause();}}}>리플레이 재생 / 일시정지</button><label>배속 <select aria-label="리플레이 배속" value={rate} onChange={e=>{const value=Number(e.target.value);if(video.current)video.current.playbackRate=value;setRate(value);}} className="bg-surface">{[.25,.5,1,2,4].map(n=><option key={n} value={n}>{n}×</option>)}</select></label><span>{time.toFixed(2)}초 / {(duration||0).toFixed(2)}초</span></div>
      <input type="range" aria-label="리플레이 타임라인" min={0} max={duration||0} step={1/data.fps} value={Math.min(time,duration||0)} onChange={e=>{const value=Number(e.target.value);if(video.current)video.current.currentTime=value;setTime(value);}} className="w-full"/>
      <label className="block py-2">전체 영상 경로 필터 <select aria-label="리플레이 경로 필터" value={route} onChange={e=>setRoute(e.target.value)} className="max-w-full bg-surface"><option value="">모든 관측 ID</option>{data.routes.map(r=><option key={JSON.stringify([r.from_zone,r.to_zone])} value={JSON.stringify([r.from_zone,r.to_zone])}>{r.from_zone} → {r.to_zone} ({r.transitions}회)</option>)}</select></label>
      {selectedRoute&&<p className="pb-2 text-muted">경로 필터는 마커·전환 이벤트에 적용됩니다. 구역 인원과 누적 집계는 전체 관측 기준입니다.</p>}
      {!data.coverage.complete_frame_index&&<p className="text-warn">처리 프레임 기록 미확인 — 인원을 임의로 0명 처리하지 않습니다.</p>}
      <div className="overflow-x-auto"><table className="w-full text-left"><thead><tr>{['구역','현재 관측 인원','누적 유입','누적 유출','순유입'].map(t=><th key={t} className="p-1">{t}</th>)}</tr></thead><tbody>{frame?.zones.map(z=><tr key={z.zone}><td className="p-1">{z.zone}</td><td>{metric(z.count)}</td><td>{metric(z.inflow)}{z.inflow===null&&` (확인 ${z.confirmed_inflow})`}</td><td>{metric(z.outflow)}{z.outflow===null&&` (확인 ${z.confirmed_outflow})`}</td><td>{metric(z.net_inflow)}</td></tr>)}</tbody></table></div>
      {!frame?.flow_complete&&<p className="py-2 text-warn">경로 미확인·단절 가능성이 있어 총 유입/유출은 미분석입니다. 괄호는 확인된 이벤트 수로 실제 총인원의 하한입니다.</p>}
      {recentIssues.length>0&&<p className="py-1 text-warn">{[...new Set(recentIssues.map(d=>diagnosticLabels[d.reason]??d.reason))].join(' · ')} — 해당 경로를 연결하지 않습니다.</p>}
      <p className="py-2">초별 평균 관측 인원: {currentMetrics.length?currentMetrics.map(r=>`${r.region_key} ${metric(r.mean_person_count)}`).join(' / '):'미분석'} · 실제 밀도(명/m²): 보정 없음</p>
      <p className="text-warn">현재 병목: {currentScores.length?currentScores.map(s=>`${s.region_key} ${s.score} / ${s.risk_level}`).join(' · '):summary?.analysis_profile==='fast'?'미분석 (FAST)':'미분석 (필수 지표 또는 기준 창 부족)'}</p>
      <details className="mt-3"><summary>경로 전환 횟수와 관측 시작→끝 OD</summary><div className="max-h-48 overflow-auto py-2"><table className="w-full text-left"><thead><tr><th>출발</th><th>도착</th><th>현재까지 확인</th><th>전체 확인</th></tr></thead><tbody>{data.routes.map(r=><tr key={r.from_zone+'→'+r.to_zone}><td>{r.from_zone}</td><td>{r.to_zone}</td><td>{(data.route_prefix_counts?.find(v=>v.from_zone===r.from_zone&&v.to_zone===r.to_zone)?.transitions??0)+data.events.filter(e=>e.from_zone===r.from_zone&&e.to_zone===r.to_zone&&e.timestamp_ms<=time*1000).length}회</td><td>{r.transitions}회</td></tr>)}</tbody></table>{!data.routes.length&&<p>확인된 구역 경로 전환 없음 — 전체 이동이 0명이라는 판정은 아닙니다.</p>}<p className="py-2 text-muted">OD는 단절 없는 관측 ID당 시작→끝 1건입니다. 실제 여행 출발·도착 또는 고유 방문자 수가 아닙니다.</p>{data.observed_od.map(r=><p key={r.from_zone+'→'+r.to_zone}>관측 OD {r.from_zone} → {r.to_zone}: {r.tracks}개 궤적</p>)}{!data.observed_od.length&&<p>집계 가능한 관측 OD 없음 (경로 단절 등으로 미분석 가능)</p>}</div></details>
      <details className="mt-2"><summary>현재 시각 이전의 확인된 전환 이벤트</summary><div className="max-h-48 overflow-auto">{[...(data.earlier_events??[]),...data.events].filter(e=>e.timestamp_ms<=time*1000&&(!selectedRoute||e.from_zone===selectedRoute[0]&&e.to_zone===selectedRoute[1])).slice(-30).map(e=><p key={e.event_id}>ID {e.track_id} · {e.from_zone||'구역 밖'} → {e.to_zone||'구역 밖'} · {(e.timestamp_ms/1000).toFixed(2)}초 · 신뢰도 {e.confidence.toFixed(2)} (미보정){e.via_outside?' · 관측된 구역 밖 경유':''}</p>)}</div></details>
    </>}
  </section>;
}
