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
