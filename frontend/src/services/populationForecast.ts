import {api} from './api';
export type PopulationModel='lightgbm'|'lstm';
export interface ForecastCatalog {
  status:'ready'|'unavailable';code:string;message:string;areas:string[];models:Partial<Record<PopulationModel,string>>;
  missing_dependencies?:string[];lookback_hours?:number;horizon_hours?:number;manifest_sha256?:string;
  model_state?:'not_loaded'|'loading'|'loaded'|'failed';last_error_code?:string|null;
}
export interface PopulationPrediction {
  area_id:string;model:PopulationModel;model_version:string;last_observed_at:string;target_at:string;
  current_value:number|null;value:number|null;target_actual:number|null;status:string;change:number|null;change_percent:number|null;
}
export interface PopulationResult {
  kind:'administrative_population_estimate_forecast';timezone:string;horizon_hours:number;manifest_sha256:string;
  items:PopulationPrediction[];cache_hit:boolean;
  info:{input_is_live:false;imputed_cells:number;still_missing_cells:number;source:string;comparison:string};
}
export async function getForecastCatalog(signal?:AbortSignal){
  return (await api.get<ForecastCatalog>('/api/population-forecast/catalog',{signal})).data;
}
export async function uploadPopulationHistory(file:File,signal?:AbortSignal){
  const form=new FormData();form.append('file',file);
  return (await api.post<PopulationResult>('/api/population-forecast/upload',form,{signal,timeout:120000})).data;
}
