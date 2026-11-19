import {useEffect,useState} from 'react';
import {api} from '../../services/api';

export default function PlaybackCompatibility({videoId,onReady}:{videoId:string;onReady:()=>void}){
  const [message,setMessage]=useState('브라우저에서 원본 영상을 재생할 수 없습니다. 코덱 또는 파일 상태를 확인하세요.');
  const [busy,setBusy]=useState(false);
  useEffect(()=>{const c=new AbortController();void api.get(`/api/videos/${encodeURIComponent(videoId)}/media-info`,{signal:c.signal}).then(r=>{
    if(c.signal.aborted)return;
    if(r.data.playback.status==='ready'){onReady();return;}
    setMessage(`원본 코덱: ${r.data.source.codec_name}. H.264 재생용 복사본을 만들 수 있습니다. 원본 분석 영상은 보존됩니다.`);
  }).catch(()=>{if(!c.signal.aborted)setMessage('원본 영상 재생 실패. FFmpeg 설정 및 영상 파일 상태를 확인하세요.');});return()=>c.abort();},[videoId,onReady]);
  useEffect(()=>{
    if(!busy)return;const c=new AbortController();let timer:ReturnType<typeof setTimeout>;
    async function poll(){try{const r=await api.get(`/api/videos/${encodeURIComponent(videoId)}/media-info`,{signal:c.signal});if(c.signal.aborted)return;
      const p=r.data.playback;
      if(p.status==='ready'){setBusy(false);onReady();return;}
      if(p.status==='failed'){setBusy(false);setMessage(`재생용 변환 실패: ${p.error}`);return;}
      timer=setTimeout(()=>void poll(),1500);
    }catch{if(!c.signal.aborted){setBusy(false);setMessage('변환 상태 조회 실패. 다시 시도하세요.');}}}
    timer=setTimeout(()=>void poll(),500);return()=>{c.abort();clearTimeout(timer);};
  },[busy,videoId,onReady]);
  async function convert(){setBusy(true);setMessage('H.264 변환 및 전체 프레임 시간축 검증 중…');try{await api.post(`/api/videos/${encodeURIComponent(videoId)}/playback`);}catch{setBusy(false);setMessage('변환 요청 실패. FFmpeg 설정, 여유 공간 및 변환 작업 상태를 확인하세요.');}}
  return <div role="alert" className="space-y-1 text-xs text-warn"><p>{message}</p><button type="button" disabled={busy} className="rounded-sm border border-line px-2 py-1" onClick={()=>void convert()}>{busy?'재생용 변환 중…':'H.264 재생용 변환'}</button></div>;
}
