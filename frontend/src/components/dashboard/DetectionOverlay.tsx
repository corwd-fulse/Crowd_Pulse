import {useEffect,useState} from 'react';
import type {RefObject} from 'react';
import {getDetections} from '../../services/api';
import type {DetectionResult} from '../../services/api';
import {detectionAt} from '../../services/detectionTiming';

export default function DetectionOverlay({jobId,videoId,video,resolution,show,emptyMessage='완료된 분석 작업이 없습니다.'}: {jobId:string|null;videoId:string;video:RefObject<HTMLVideoElement|null>;resolution:[number,number];show:boolean;emptyMessage?:string}) {
  const [stored,setStored]=useState<{id:string;data:DetectionResult|null;error?:boolean}|null>(null);
  const [time,setTime]=useState(0);
  const [displayWidth,setDisplayWidth]=useState(960);
  useEffect(()=>{
    const element=video.current;
    if(!element||typeof ResizeObserver==='undefined')return;
    const observer=new ResizeObserver(entries=>{const width=entries[0].contentRect.width;if(width>0)setDisplayWidth(width);});
    observer.observe(element);return ()=>observer.disconnect();
  },[video]);
  useEffect(()=>{
    if(!jobId)return;
    const controller=new AbortController();
    getDetections(jobId,controller.signal).then(data=>{
      if(data&&(data.analysis_job_id!==jobId||data.video_id.replaceAll('-','')!==videoId.replaceAll('-','')))throw new Error('Mismatched tracking result');
      if(!controller.signal.aborted)setStored({id:jobId,data});
    }).catch(()=>{if(!controller.signal.aborted)setStored({id:jobId,data:null,error:true});});
    return ()=>controller.abort();
  },[jobId,videoId]);
  useEffect(()=>{
    const element=video.current;if(!element)return;
    let stopped=false,callback=0;
    const update=()=>setTime(element.currentTime);
    const fallbackUpdate=()=>{if(element.paused||!element.requestVideoFrameCallback)update();};
    const tick:VideoFrameRequestCallback=(_,metadata)=>{if(stopped)return;setTime(metadata.mediaTime);callback=element.requestVideoFrameCallback(tick);};
    if(element.requestVideoFrameCallback)callback=element.requestVideoFrameCallback(tick);
    element.addEventListener('seeked',update);element.addEventListener('loadedmetadata',update);element.addEventListener('timeupdate',fallbackUpdate);
    return ()=>{stopped=true;if(callback)element.cancelVideoFrameCallback(callback);element.removeEventListener('seeked',update);element.removeEventListener('loadedmetadata',update);element.removeEventListener('timeupdate',fallbackUpdate);};
  },[video]);
  const state=stored?.id===jobId?stored:null;
  const mismatch=state?.data&&state.data.video_resolution.some((n,i)=>n!==resolution[i]);
  const frame=state?.data&&!mismatch?detectionAt(state.data,time):undefined;
  return <>
    {show&&state?.data&&!mismatch&&<svg aria-label="YOLO 탐지 박스" data-frame={frame?.frame??''} viewBox={`0 0 ${resolution[0]} ${resolution[1]}`} className="pointer-events-none absolute inset-0 h-full w-full">{frame?.boxes.map((box,i)=><g key={i}><rect x={box.x1} y={box.y1} width={box.x2-box.x1} height={box.y2-box.y1} fill="none" stroke="#23e3bb" strokeWidth={2} vectorEffect="non-scaling-stroke"/><text x={box.x1+2} y={Math.max(12*resolution[0]/displayWidth,box.y1-5)} fontSize={12*resolution[0]/displayWidth} fill="#23e3bb" stroke="#061526" strokeWidth={2*resolution[0]/displayWidth} paintOrder="stroke">{box.track_id===null?'ID 없음':`ID ${box.track_id}`}</text></g>)}</svg>}
    <div className="pointer-events-none absolute left-2 top-2 max-w-[90%] rounded-sm bg-[#061526]/90 px-2 py-1 text-xs text-white" role="status">
      {!jobId?emptyMessage:!state?'탐지 CSV를 불러오는 중…':state.error?'탐지 CSV 로딩·검증 오류입니다.':!state.data?'이 작업의 탐지 CSV가 없습니다.':mismatch?'탐지 결과와 영상 해상도가 다릅니다.':!show?'YOLO 표시 꺼짐':!state.data.frames.length?'탐지 CSV는 정상이며 탐지 행이 없습니다.':`프레임 ${Math.floor(time*state.data.fps+1e-6)} · ${frame?.boxes.length??0}개 탐지 · ${state.data.fps} FPS`}
    </div>
  </>;
}
