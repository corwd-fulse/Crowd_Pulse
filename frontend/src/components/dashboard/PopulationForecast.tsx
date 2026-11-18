import {useEffect,useRef,useState} from 'react';
import axios from 'axios';
import {getForecastCatalog,uploadPopulationHistory} from '../../services/populationForecast';
import type {ForecastCatalog,PopulationModel,PopulationResult} from '../../services/populationForecast';

const reasons:Record<string,string>={insufficient_history:'이력 부족 (최소 168시간 필요)',gap_in_history:'최근 이력 결측',unknown_area:'학습 범위 밖 지역',invalid_prediction:'예측 출력 확인 불가'};
const number=(value:number|null)=>value===null?'미분석':value.toLocaleString('ko-KR',{maximumFractionDigits:1});
const time=(value:string)=>new Date(value).toLocaleString('ko-KR',{timeZone:'Asia/Seoul'});

export default function PopulationForecast(){
  const [catalog,setCatalog]=useState<ForecastCatalog|null>(null),[error,setError]=useState('');
  const [file,setFile]=useState<File|null>(null),[busy,setBusy]=useState(false);
  const [result,setResult]=useState<PopulationResult|null>(null);
  const [area,setArea]=useState(''),[model,setModel]=useState<PopulationModel>('lightgbm');
  const request=useRef<AbortController|null>(null);
  useEffect(()=>{
    const controller=new AbortController();
    getForecastCatalog(controller.signal).then(data=>{if(!controller.signal.aborted){setCatalog(data);setArea(data.areas[0]??'');}})
      .catch(()=>{if(!controller.signal.aborted)setError('생활인구 예측 API 상태를 확인할 수 없습니다. 예측값을 표시하지 않습니다.');});
    return()=>{controller.abort();request.current?.abort();};
  },[]);
  async function predict(){
    if(!file||busy)return;
    if(file.size>10*1024*1024){setError('입력 파일은 10MiB 이하여야 합니다.');return;}
    const controller=new AbortController();request.current=controller;setBusy(true);setError('');setResult(null);
    try{
      const data=await uploadPopulationHistory(file,controller.signal);
      if(controller.signal.aborted)return;
      setResult(data);setArea(data.items[0]?.area_id??'');
    }catch(error){
      if(controller.signal.aborted)return;
      const detail=axios.isAxiosError(error)?error.response?.data?.detail:undefined;
      setError(typeof detail?.message==='string'?detail.message:'생활인구 예측에 실패했습니다. 결측을 0명으로 대체하지 않습니다.');
    }finally{if(!controller.signal.aborted){setBusy(false);void getForecastCatalog(controller.signal).then(setCatalog).catch(()=>{});}}
  }
  const areas=result?[...new Set(result.items.map(row=>row.area_id))]:catalog?.areas??[];
  const selected=result?.items.find(row=>row.area_id===area&&row.model===model);
  return <section aria-label="행정동 생활인구 예측" data-source="population-forecast" className="min-w-0 space-y-4 rounded-lg border border-line bg-surface p-4 sm:p-5">
    <div><h2 className="font-bold text-ink">행정동 생활인구 · 1시간 후 예측</h2><p className="mt-2 text-xs text-muted">서울 행정동 단위 생활인구 추정치의 시계열 예측입니다. 영상 ROI 인원·2D/3D 관측 리플레이와 별도이며, 현장 실측이나 실시간 인구가 아닙니다.</p></div>
    {!catalog&&!error&&<p role="status">예측 모델 연결 상태 확인 중…</p>}
    <p className="text-xs text-muted">모델 로드 상태: {busy?'로드·추론 중':catalog?.model_state==='loaded'?'로드 완료':catalog?.model_state==='loading'?'로딩 중':catalog?.model_state==='failed'?'실행 실패':catalog?.model_state==='not_loaded'?'아직 로드하지 않음':'미확인'}</p>
    {catalog?.status==='unavailable'&&<div role="status" className="text-sm text-warn"><p>{catalog.message}</p>{!!catalog.missing_dependencies?.length&&<p className="mt-1">필수 의존성 미확인: {catalog.missing_dependencies.join(', ')}</p>}<p>연결 준비 전에는 예측값을 표시하지 않습니다.</p></div>}
    <p className="text-xs text-muted">CSV/ZIP 업로드: 최소 168시간 연속 이력, 169시간 권장. 최대 10MiB·100,000행. CSV는 observed_at, area_id, estimate 또는 서울 생활인구 원본 앞 4열입니다. 시간대 없는 시각은 한국 시각으로 해석합니다. 지역 이름 매핑이 없어 8자리 코드로 표시합니다.</p>
    <div className="flex flex-wrap items-end gap-3 text-xs">
      <label className="min-w-0">생활인구 이력 파일<input aria-label="생활인구 이력 파일" type="file" accept=".csv,.zip" disabled={busy} onChange={e=>{setFile(e.target.files?.[0]??null);setResult(null);setError('');}} className="mt-2 block max-w-full"/></label>
      <button type="button" disabled={busy||!file||catalog?.status!=='ready'} onClick={()=>void predict()} className="rounded-sm border border-brand px-4 py-2 text-brand disabled:opacity-40">{busy?'생활인구 예측 중…':'1시간 후 생활인구 예측'}</button>
      <label>모델 <select aria-label="생활인구 예측 모델" value={model} onChange={e=>setModel(e.target.value as PopulationModel)} className="rounded-sm border border-line bg-surface p-2"><option value="lightgbm">LightGBM</option><option value="lstm">LSTM</option></select></label>
      <label>행정동 코드 <select aria-label="생활인구 예측 지역" value={area} onChange={e=>setArea(e.target.value)} className="rounded-sm border border-line bg-surface p-2">{areas.length?areas.map(id=><option key={id} value={id}>{id}</option>):<option value="">입력 지역 없음</option>}</select></label>
    </div>
    {error&&<p role="alert" className="text-sm text-warn">{error}</p>}
    {selected&&<div className="space-y-3 text-sm" data-forecast-status={selected.status}>
      <p className="text-xs text-muted">기준 시각: {time(selected.last_observed_at)} KST → 예측 대상: {time(selected.target_at)} KST · 지역 {selected.area_id} · {selected.model_version}</p>
      {selected.status!=='ok'&&<p className="text-warn">미분석: {reasons[selected.status]??selected.status}</p>}
      <dl className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="rounded-sm border border-line p-3"><dt className="text-xs text-muted">마지막 입력 생활인구 추정치</dt><dd className="mt-2 text-lg text-ink">{number(selected.current_value)}{selected.current_value!==null&&'명'}</dd></div>
        <div className="rounded-sm border border-brand p-3"><dt className="text-xs text-muted">1시간 후 모델 예측 (관측값 아님)</dt><dd className="mt-2 text-lg text-brand">{number(selected.value)}{selected.value!==null&&'명'}</dd></div>
        <div className="rounded-sm border border-line p-3"><dt className="text-xs text-muted">입력 대비 예상 변화</dt><dd className="mt-2 text-lg">{number(selected.change)}{selected.change!==null&&'명'} · {selected.change_percent===null?'변화율 계산 불가':`${selected.change_percent.toFixed(2)}%`}</dd></div>
      </dl>
      <p className="text-xs text-muted">업로드 이력 기준 예측이며 실시간 데이터 연결이 아닙니다. 중간 이력 보간 {result?.info.imputed_cells??'미확인'}개 · 남은 결측 {result?.info.still_missing_cells??'미확인'}개 · {result?.cache_hit?'캐시 결과 재사용':'모델 추론 결과'}</p>
    </div>}
    {result&&!selected&&<p className="text-warn">선택 지역·모델의 예측 결과가 없습니다.</p>}
  </section>;
}
