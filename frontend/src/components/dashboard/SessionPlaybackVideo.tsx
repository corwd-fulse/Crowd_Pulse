import {useCallback,useState} from 'react';
import {api} from '../../services/api';
import PlaybackCompatibility from './PlaybackCompatibility';

export default function SessionPlaybackVideo({videoId,controls,register,offset}:{videoId:string;controls:boolean;register:(v:HTMLVideoElement|null)=>void;offset:number}){
  const [copy,setCopy]=useState(false),[failed,setFailed]=useState(false);
  const ready=useCallback(()=>{setCopy(true);setFailed(false);},[]);
  return <><video className="w-full" muted controls={controls} preload="metadata" ref={register}
    src={api.getUri({url:`/api/videos/${encodeURIComponent(videoId)}/${copy?'playback/stream':'stream'}`})}
    onLoadedMetadata={e=>{if(offset>=0&&offset<e.currentTarget.duration)e.currentTarget.currentTime=offset;setFailed(false);}}
    onError={()=>setFailed(true)}/>{failed&&<PlaybackCompatibility videoId={videoId} onReady={ready}/>}</>;
}
