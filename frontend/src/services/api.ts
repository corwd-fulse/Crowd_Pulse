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

export async function getAnalysisJobsByVideo(videoId: string): Promise<AnalysisJobResponse[]> {
  const response = await api.get<AnalysisJobResponse[]>('/api/analysis/jobs', {
    params: {video_id: videoId},
  });
  return response.data;
}

export async function createAnalysisJob(videoId: string): Promise<AnalysisJobResponse> {
  const response = await api.post<AnalysisJobResponse>('/api/analysis/jobs', {
    video_id: videoId,
  });
  return response.data;
}

export async function getAnalysisJob(jobId: string): Promise<AnalysisJobResponse> {
  const response = await api.get<AnalysisJobResponse>(
    `/api/analysis/jobs/${encodeURIComponent(jobId)}`
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
