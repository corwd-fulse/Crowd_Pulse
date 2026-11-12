import axios from 'axios';

export const api = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
    timeout: 10000, // 10초
})


export interface HealthResponse {
    status: string;
    service: string;
    version: string;
}

export async function getHealth(): Promise<HealthResponse> {
    const response = await api.get<HealthResponse>('/health');
    return response.data;
}


export interface DashboardSummary {
  person_count: number;
  congestion_percent: number;
  bottleneck_count: number;
  alert_count: number;
  updated_at: string;
  data_source: string;
}

export async function getDashboardSummary(): Promise<DashboardSummary> {
  const response = await api.get<DashboardSummary>(
    '/api/dashboard/summary'
  );

  return response.data;
}


export interface VideoUploadResponse {
  video_id: string;
  filename: string;
  saved_filename: string;
  file_size: number;
  status: string;
  data_source: string;
}

export interface AnalysisJobResponse {
  id: string;
  video_id: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';
  model_version: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
  analysis_mode: 'production' | 'test';
  analysis_profile?: 'fast' | 'full';
  roi_configured?: boolean;
}

export async function getAnalysisCapabilities(signal?:AbortSignal):Promise<{test_analysis_enabled:boolean;execution:string;auto_start_worker:boolean}> {
  return (await api.get('/api/analysis/capabilities',{signal})).data;
}

export async function getAnalysisJobs(signal?: AbortSignal): Promise<AnalysisJobResponse[]> {
  const response = await api.get<AnalysisJobResponse[]>('/api/analysis/jobs', {signal});
  return response.data;
}

export async function getLatestCompletedJob(signal?: AbortSignal): Promise<AnalysisJobResponse | null> {
  const response=await api.get<AnalysisJobResponse[]>('/api/analysis/jobs',{
    signal,params:{status:'succeeded',order_by:'finished_at',limit:1},
  });
  return response.data[0]??null;
}

export interface DirectionPresentation {
  quality: 'unvalidated' | 'validated' | 'rejected';
  confirmed_for_display: boolean;
  label: string;
  algorithm_version: string;
  csv_schema_version: string;
}

export interface RoiSnapshot {
  status: string;
  approved_for_analysis: boolean;
  video_resolution: [number, number];
  zones: Array<{name: string; polygon: [number, number][]; description?: string}>;
  [key: string]: unknown;
}

export interface DetectionBox {track_id:number|null;x1:number;y1:number;x2:number;y2:number;confidence:number}
export interface DetectionFrame {frame:number;t_ms:number;boxes:DetectionBox[]}
export interface DetectionResult {
  analysis_job_id:string;video_id:string;fps:number;frame_count:number;video_resolution:[number,number];frames:DetectionFrame[];
}
export async function getDetections(jobId:string,signal?:AbortSignal):Promise<DetectionResult|null> {
  try {return (await api.get<DetectionResult>(`/api/analysis/jobs/${encodeURIComponent(jobId)}/detections`,{signal})).data;}
  catch(error) {if(axios.isAxiosError(error)&&error.response?.status===404)return null;throw error;}
}

export interface VisionResultSummary {
  analysis_job_id: string;
  video_id: string;
  status: string;
  analysis_mode: 'production' | 'test';
  analysis_profile?: 'fast' | 'full';
  components?: Record<'tracking' | 'observations' | 'individual_directions' | 'optical_flow' | 'bottleneck', 'completed' | 'not_run'>;
  roi_status: 'candidate' | 'approved';
  roi_snapshot: RoiSnapshot;
  video_resolution: [number, number];
  execution_snapshot: {
    algorithm_version?: string;
    model_version?: string;
    direction_quality?: DirectionPresentation['quality'];
    settings?: Record<string, unknown>;
    [key: string]: unknown;
  };
  analysis_started_at: string;
  analysis_finished_at: string;
  imported_at: string;
  artifacts: Record<string, {rows: number; bytes: number; sha256: string; missing: Record<string, number>; download_url: string}>;
  direction_presentation: DirectionPresentation;
  units: Record<string, string>;
}

export interface VisionZoneMetric {
  analysis_job_id: string;
  imported_at: string;
  region_key: string;
  video_offset_ms: number;
  has_observation: boolean;
  has_flow: boolean;
  has_individual_aggregate: boolean;
  mean_person_count: number | null;
  inflow: number | null;
  outflow: number | null;
  flow_angle_deg: number | null;
  magnitude_px: number | null;
  motion: number | null;
  low_confidence: boolean | null;
  mean_angle_deg: number | null;
  direction_sample_count: number | null;
  flow_direction_confirmed: boolean;
  individual_direction_confirmed: boolean;
}

export interface VisionBottleneckScore {
  analysis_job_id: string;
  imported_at: string;
  region_key: string;
  window_start_ms: number;
  score: number;
  growth: number;
  slowdown: number;
  imbalance: number;
  risk_level: '낮음' | '주의' | '높음';
  severity: 'low' | 'medium' | 'high';
}

export interface ResultPage<T> {
  items: T[];
  limit: number;
  offset: number;
}

export interface VisionResultBundle {
  summary: VisionResultSummary;
  metrics: VisionZoneMetric[];
  scores: VisionBottleneckScore[];
}

export async function getVisionResult(jobId: string, signal?: AbortSignal): Promise<VisionResultSummary> {
  return (await api.get<VisionResultSummary>(`/api/analysis/jobs/${encodeURIComponent(jobId)}/results`, {signal})).data;
}

export async function getVisionZoneMetrics(jobId: string, offset = 0, signal?: AbortSignal): Promise<ResultPage<VisionZoneMetric> & {direction_presentation: DirectionPresentation}> {
  return (await api.get(`/api/analysis/jobs/${encodeURIComponent(jobId)}/results/zone-metrics`, {params: {limit: 1000, offset}, signal})).data;
}

export async function getVisionBottleneckScores(jobId: string, offset = 0, signal?: AbortSignal): Promise<ResultPage<VisionBottleneckScore>> {
  return (await api.get(`/api/analysis/jobs/${encodeURIComponent(jobId)}/results/bottleneck-scores`, {params: {limit: 1000, offset}, signal})).data;
}

async function allPages<T>(fetchPage: (offset: number) => Promise<ResultPage<T>>): Promise<T[]> {
  const items: T[] = [];
  for (;;) {
    const page = await fetchPage(items.length);
    if (page.offset !== items.length || page.limit <= 0 || page.items.length > page.limit) {
      throw new Error('결과 페이지 응답이 올바르지 않습니다.');
    }
    items.push(...page.items);
    if (page.items.length < page.limit) return items;
  }
}

export async function getVisionResultBundle(jobId: string, signal?: AbortSignal): Promise<VisionResultBundle> {
  const [summary, metrics, scores] = await Promise.all([
    getVisionResult(jobId, signal),
    allPages((offset) => getVisionZoneMetrics(jobId, offset, signal)),
    allPages((offset) => getVisionBottleneckScores(jobId, offset, signal)),
  ]);
  if (summary.analysis_job_id !== jobId || [...metrics, ...scores].some((row) => row.analysis_job_id !== jobId)) {
    throw new Error('다른 작업의 결과가 반환되었습니다. 다시 조회하세요.');
  }
  return {summary, metrics, scores};
}

export async function getVisionObservations(jobId:string,signal?:AbortSignal):Promise<Pick<VisionResultBundle,'summary'|'metrics'>> {
  const [summary,metrics]=await Promise.all([getVisionResult(jobId,signal),allPages(offset=>getVisionZoneMetrics(jobId,offset,signal))]);
  if(summary.analysis_job_id!==jobId||metrics.some(row=>row.analysis_job_id!==jobId))throw new Error('Mismatched observation job');
  return {summary,metrics};
}

export interface VideoListItem {
  id: string;
  camera_id: number | null;
  filename: string;
  saved_filename: string;
  file_size: number;
  content_type: string | null;
  duration_ms: number | null;
  recorded_at: string | null;
  uploaded_at: string;
  status: string;
  data_source: string;
}

export async function getVideos(limit = 50, offset = 0): Promise<VideoListItem[]> {
  const response = await api.get<VideoListItem[]>('/api/videos', {
    params: {limit, offset},
  });
  return response.data;
}

export async function getVideoDetails(videoId:string,signal?:AbortSignal):Promise<Pick<VideoListItem,'id'|'filename'>> {
  return (await api.get<Pick<VideoListItem,'id'|'filename'>>(`/api/videos/${encodeURIComponent(videoId)}`,{signal})).data;
}

export async function getAnalysisJobsByVideo(videoId: string): Promise<AnalysisJobResponse[]> {
  const response = await api.get<AnalysisJobResponse[]>('/api/analysis/jobs', {
    params: {video_id: videoId},
  });
  return response.data;
}

export async function createAnalysisJob(videoId: string, mode: 'production' | 'test' = 'production', profile: 'fast' | 'full' = 'full'): Promise<AnalysisJobResponse> {
  const response = await api.post<AnalysisJobResponse>('/api/analysis/jobs', {
    video_id: videoId,
    analysis_profile: profile,
    ...(mode === 'test' ? {analysis_mode: mode} : {}),
  });
  return response.data;
}

export interface VideoRoiResponse {video_id: string; roi_snapshot: RoiSnapshot; roi_status?: 'candidate' | 'approved'}
export async function getVideoRoi(videoId: string, signal?: AbortSignal): Promise<VideoRoiResponse | null> {
  try {
    return (await api.get<VideoRoiResponse>(`/api/videos/${encodeURIComponent(videoId)}/roi`, {signal})).data;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response?.status === 404) return null;
    throw error;
  }
}
export async function registerVideoRoi(videoId: string, snapshot: RoiSnapshot): Promise<VideoRoiResponse> {
  return (await api.put<VideoRoiResponse>(`/api/videos/${encodeURIComponent(videoId)}/roi`, snapshot,
    {headers: {'If-None-Match': '*'}})).data;
}

export async function getAnalysisJob(jobId: string, signal?: AbortSignal): Promise<AnalysisJobResponse> {
  const response = await api.get<AnalysisJobResponse>(
    `/api/analysis/jobs/${encodeURIComponent(jobId)}`, {signal}
  );
  return response.data;
}

export async function uploadVideo(
  file: File
): Promise<VideoUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<VideoUploadResponse>(
    '/api/videos/upload',
    formData,
    {
      timeout: 120000,
    }
  );

  return response.data;
}
