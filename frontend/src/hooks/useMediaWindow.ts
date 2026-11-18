import {useEffect,useRef,useState} from 'react';
export type MediaWindow={window?:{start_ms:number;end_ms:number;next_start_ms:number|null};frame_count:number;fps:number};
type Fetch<T>=(id:string,signal?:AbortSignal,start?:number,end?:number)=>Promise<T|null>;
const covers=(data:MediaWindow,ms:number)=>!data.window||ms>=data.window.start_ms&&ms<data.window.end_ms;

export function useMediaWindow<T extends MediaWindow>(jobId:string|null,videoId:string,time:number,fetch:Fetch<T>){
  const [data,setData]=useState<T|null>(null),[error,setError]=useState(false),[loaded,setLoaded]=useState(false);
  const generation=useRef(0),pending=useRef(new Set<string>()),next=useRef<T|null>(null),clock=useRef(time*1000);
  const missing=useRef(false),retryAfter=useRef(0);
  clock.current=time*1000;
  useEffect(()=>{generation.current++;pending.current.clear();next.current=null;missing.current=false;retryAfter.current=0;setData(null);setError(false);setLoaded(false);return()=>{generation.current++;};},[jobId,videoId]);
  useEffect(()=>{
    if(!jobId||missing.current||Date.now()<retryAfter.current)return;const version=generation.current,ms=time*1000;
    function playbackTime(value:number,basis:T|null){
      if(!basis)return value;
      // The browser can report the exact duration on the final timeupdate.
      const duration=basis.frame_count/basis.fps*1000;
      return value<duration?value:duration-500/basis.fps;
    }
    const target=playbackTime(ms,data);
    function startFor(value:number,basis:T|null=data){
      const base=Math.floor(value/5000)*5000;
      return basis?.window&&value>=basis.window.end_ms&&base<basis.window.end_ms
        ? value : base;
    }
    function load(start:number,prefetch=false){
      const key=String(start);if(pending.current.has(key))return;pending.current.add(key);
      void fetch(jobId!,new AbortController().signal,Math.max(0,start),Math.max(0,start)+5000).then(result=>{
        if(generation.current!==version)return;
        setLoaded(true);
        if(!result){missing.current=true;setData(null);return;}
        const scoped=result as T&{analysis_job_id?:string;video_id?:string};
        if(scoped.analysis_job_id&&scoped.analysis_job_id!==jobId||scoped.video_id&&scoped.video_id.replaceAll('-','')!==videoId.replaceAll('-',''))throw new Error('Media source mismatch');
        setError(false);
        const current=playbackTime(clock.current,result);
        if(covers(result,current))setData(result);
        else if(prefetch)next.current=result;
        else {
          const targetStart=startFor(current,result);
          if(targetStart===start)throw new Error('Replay window cannot reach the selected time');
          load(targetStart);
        }
      }).catch(()=>{if(generation.current===version){retryAfter.current=Date.now()+3000;if(!prefetch)setError(true);}}).finally(()=>{if(generation.current===version)pending.current.delete(key);});
    }
    if(data&&covers(data,target)){
      if(data.window&&data.window.next_start_ms!==null&&data.window.end_ms-target<2000&&(!next.current||!covers(next.current,data.window.next_start_ms)))load(data.window.next_start_ms,true);
      return;
    }
    if(next.current&&covers(next.current,target)){setData(next.current);next.current=null;return;}
    next.current=null;load(startFor(target));
  },[jobId,videoId,time,data,fetch]);
  return {data:data&&covers(data,time*1000)?data:null,error,loaded};
}
