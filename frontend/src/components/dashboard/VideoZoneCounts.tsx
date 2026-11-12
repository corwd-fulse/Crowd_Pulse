import {useEffect,useState} from 'react';
import type {RefObject} from 'react';
import {getVisionObservations} from '../../services/api';
import type {VisionResultBundle} from '../../services/api';

export default function VideoZoneCounts({jobId,videoId,video}: {jobId:string|null;videoId:string;video:RefObject<HTMLVideoElement|null>}) {
  const [stored,setStored]=useState<{id:string;data?:Pick<VisionResultBundle,'summary'|'metrics'>;error?:boolean}|null>(null);
  const [second,setSecond]=useState(0);
  useEffect(()=>{
    if(!jobId)return;
    const controller=new AbortController();
    getVisionObservations(jobId,controller.signal).then(data=>{
      if(data.summary.analysis_job_id!==jobId||data.metrics.some(row=>row.analysis_job_id!==jobId)||data.summary.video_id.replaceAll('-','')!==videoId.replaceAll('-',''))throw new Error('Mismatched observation source');
      if(!controller.signal.aborted)setStored({id:jobId,data});
    }).catch(()=>{if(!controller.signal.aborted)setStored({id:jobId,error:true});});
    return ()=>controller.abort();
  },[jobId,videoId]);
  useEffect(()=>{
    const v=video.current;if(!v)return;
    const update=()=>setSecond(Math.floor(v.currentTime));
    v.addEventListener('timeupdate',update);v.addEventListener('seeked',update);
    return ()=>{v.removeEventListener('timeupdate',update);v.removeEventListener('seeked',update);};
  },[video]);
  if(!jobId)return null;
  const state=stored?.id===jobId?stored:null;
  const data=state?.data;
  const names=data?[...new Set([...data.summary.roi_snapshot.zones.map(z=>z.name),...data.metrics.map(row=>row.region_key)])]:[];
  return <section aria-label="재생 시점 구역 인원" data-source="zone-observations" data-second={second} className="space-y-2 border-t border-line px-4 py-3 text-xs text-ink-soft">
    <h3 className="font-semibold">영상 {second}–{second+1}초 · 구역별 초당 평균 인원</h3>
    <p className="text-muted">crowd_observations 기준 소수 평균입니다. 현재 프레임의 탐지 박스 수나 고유 인원 수가 아닙니다. 분석 당시 ROI 기준입니다.</p>
    {!state?<p role="status">구역 인원 결과를 불러오는 중…</p>:state.error?<p role="alert">구역 인원 결과 조회에 실패했습니다.</p>:!data?.metrics.some(row=>row.has_observation)?<p>저장된 구역 인원 관측 결과가 없습니다.</p>:<div className="flex flex-wrap gap-3">{names.map(name=>{
      const row=data.metrics.find(row=>row.region_key===name&&row.video_offset_ms===second*1000&&row.has_observation);
      return <span key={name} data-region={name} className="rounded-sm border border-line px-3 py-2">{name}: {row?.mean_person_count==null?'데이터 없음':`${Number(row.mean_person_count).toLocaleString('ko-KR',{maximumFractionDigits:4})}명`}</span>;
    })}</div>}
  </section>;
}
