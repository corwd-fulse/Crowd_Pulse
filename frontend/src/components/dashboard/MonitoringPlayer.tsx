import {useEffect,useRef,useState} from 'react';
import {api,getVisionResult,getVideoDetails} from '../../services/api';
import type {AnalysisJobResponse,VisionResultSummary} from '../../services/api';
import DetectionOverlay from './DetectionOverlay';
import VideoZoneCounts from './VideoZoneCounts';

// Mounted by job ID: changing the selected job discards the old video and overlays.
export default function MonitoringPlayer({job}: {job:AnalysisJobResponse}) {
  const video=useRef<HTMLVideoElement>(null);
  const [resolution,setResolution]=useState<[number,number]>([0,0]);
  const [summary,setSummary]=useState<{id:string;data:VisionResultSummary}|null>(null);
  const [roiError,setRoiError]=useState(false);
  const [videoError,setVideoError]=useState(false);
  const [videoName,setVideoName]=useState<string|null>(null);
  const [showOriginal,setShowOriginal]=useState(true);
  const [showYolo,setShowYolo]=useState(true);
  const [showRoi,setShowRoi]=useState(true);
  const completed=job.status==='succeeded';
  useEffect(()=>{
    const controller=new AbortController();
    getVideoDetails(job.video_id,controller.signal).then(data=>{
      if(data.id.replaceAll('-','')!==job.video_id.replaceAll('-',''))throw new Error('Mismatched video');
      if(!controller.signal.aborted)setVideoName(data.filename);
    }).catch(()=>{if(!controller.signal.aborted)setVideoName('영상 이름 조회 실패');});
    return ()=>controller.abort();
  },[job.video_id]);
  useEffect(()=>{
    if(!completed)return;
    const controller=new AbortController();
    getVisionResult(job.id,controller.signal).then(data=>{
      if(data.analysis_job_id!==job.id||data.video_id.replaceAll('-','')!==job.video_id.replaceAll('-',''))throw new Error('Mismatched result');
      if(!controller.signal.aborted){setSummary({id:job.id,data});setRoiError(false);}
    }).catch(()=>{if(!controller.signal.aborted)setRoiError(true);});
    return ()=>controller.abort();
  },[job.id,job.video_id,completed]);
  const data=summary?.id===job.id?summary.data:null;
  const roi=data?.roi_snapshot;
  const roiMatches=roi?.video_resolution.every((n,i)=>n===resolution[i]);
  return <div aria-label="서버 분석 영상 플레이어" className="w-full min-w-0">
    <div className="flex flex-wrap items-center gap-3 border-b border-line px-4 py-2 text-xs text-ink-soft">
      <label><input type="checkbox" checked={showOriginal} onChange={e=>setShowOriginal(e.target.checked)}/> 원본 영상</label>
      <label><input type="checkbox" checked={showYolo} onChange={e=>setShowYolo(e.target.checked)}/> YOLO 탐지</label>
      <label><input type="checkbox" checked={showRoi} onChange={e=>setShowRoi(e.target.checked)}/> ROI 구역</label>
      <button type="button" className="rounded-sm border border-line px-2 py-1" onClick={()=>{const v=video.current;if(v){if(v.paused)void v.play().catch(()=>setVideoError(true));else v.pause();}}}>재생 / 일시정지</button>
    </div>
    <div className="relative w-full" style={resolution[0]?{aspectRatio:`${resolution[0]} / ${resolution[1]}`}:{}}>
      <video ref={video} aria-label="대시보드 서버 영상" src={api.getUri({url:`/api/videos/${encodeURIComponent(job.video_id)}/stream`})} controls playsInline preload="metadata" className="block h-full w-full" style={{opacity:showOriginal?1:0}} onLoadedMetadata={e=>{setResolution([e.currentTarget.videoWidth,e.currentTarget.videoHeight]);setVideoError(false);}} onError={()=>setVideoError(true)}/>
      {!!resolution[0]&&<>
        {showRoi&&roiMatches&&<svg aria-label="분석 작업 ROI" viewBox={`0 0 ${resolution[0]} ${resolution[1]}`} className="pointer-events-none absolute inset-0 h-full w-full">{roi?.zones.map((zone,i)=><polygon key={zone.name} points={zone.polygon.map(p=>p.join(',')).join(' ')} fill={i%2?'#ffc653':'#23cfff'} fillOpacity={.1} stroke={i%2?'#ffc653':'#23cfff'} strokeWidth={2} vectorEffect="non-scaling-stroke"/>)}</svg>}
        <DetectionOverlay jobId={completed?job.id:null} videoId={job.video_id} video={video} resolution={resolution} show={showYolo} emptyMessage={job.status==='running'?'AI 분석 중':job.status==='queued'?'분석 대기 중 — 별도 Worker 실행 필요':job.status==='failed'?`분석 실패: ${job.error_message??'상세 오류 없음'}`:'아직 분석하지 않은 영상입니다'}/>
      </>}
    </div>
    {!!resolution[0]&&<VideoZoneCounts key={job.id} jobId={completed?job.id:null} videoId={job.video_id} video={video}/>}
    <div className="space-y-1 break-all border-t border-line px-4 py-2 text-xs text-muted">
      <p>현재 영상: {videoName??'영상 이름 조회 중…'} · 작업 상태: {job.status} · {job.analysis_profile==='fast'?'FAST 빠른 분석':'FULL 정밀 분석'}</p>
      <p>서버 영상 ID {job.video_id} · 작업 {job.id}</p>
      <p>{completed?'분석 완료 — 이 영상과 동일한 작업의 저장 결과만 표시합니다.':`미분석 또는 분석 미완료 (${job.status}) — 탐지 박스를 표시하지 않습니다.`}</p>
      {data&&<p>{data.analysis_mode==='test'?'테스트 분석':'운영 분석'} · {data.roi_status==='approved'&&roi?.approved_for_analysis?'승인 ROI':'미승인 후보 ROI'} · 방향 품질 {data.direction_presentation.quality}</p>}
      {roiError&&<p role="alert">ROI 스냅샷 조회에 실패했습니다. 탐지 CSV는 별도로 조회합니다.</p>}
      {roi&&!roiMatches&&!!resolution[0]&&<p role="alert">ROI 해상도가 영상과 달라 구역을 표시하지 않습니다.</p>}
      {videoError&&<p role="alert">서버 영상을 재생할 수 없습니다.</p>}
    </div>
  </div>;
}
