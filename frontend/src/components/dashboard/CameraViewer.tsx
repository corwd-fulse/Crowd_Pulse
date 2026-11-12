
import { useState, useEffect, useRef } from 'react';
import axios from 'axios';

import {
  uploadVideo,
  createAnalysisJob,
  getAnalysisJob,
} from '../../services/api';

import type {
  VideoUploadResponse,
  AnalysisJobResponse,
} from '../../services/api';

import {
  validateVideo,
  VIDEO_EXTENSIONS,
} from '../../services/videoUpload';


const STATUS_LABELS: Record<AnalysisJobResponse['status'], string> = {
  queued: '분석 대기 중',
  running: '분석 진행 중',
  succeeded: '분석 완료',
  failed: '분석 실패',
  cancelled: '분석 취소',
};


export default function CameraViewer() {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState('');

  const [isUploading, setIsUploading] = useState(false);
  const [isRequestingAnalysis, setIsRequestingAnalysis] = useState(false);

  const [error, setError] = useState('');
  const [analysisError, setAnalysisError] = useState('');

  const [result, setResult] = useState<VideoUploadResponse | null>(null);
  const [analysisJob, setAnalysisJob] = useState<AnalysisJobResponse | null>(null);

  const uploadInProgress = useRef(false);
  const analysisInProgress = useRef(false);
  const requestVersion = useRef(0);

  useEffect(() => {
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);


  // =========================
  // 영상 업로드
  // =========================

  const handleVideoUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    event.target.value = '';

    if (!file || uploadInProgress.current || analysisInProgress.current) {
      return;
    }

    const validationError = validateVideo(file);

    setError(validationError ?? '');

    if (validationError) {
      return;
    }

    // 이전 분석 결과 무효화
    const version = ++requestVersion.current;

    setResult(null);
    setAnalysisJob(null);
    setAnalysisError('');

    const url = URL.createObjectURL(file);

    setVideoUrl(url);
    setFileName(file.name);

    uploadInProgress.current = true;
    setIsUploading(true);

    try {
      const response = await uploadVideo(file);

      if (version === requestVersion.current) {
        setResult(response);
      }
    } catch (error: unknown) {
      if (version !== requestVersion.current) return;

      if (axios.isAxiosError(error)) {
        const detail: unknown = error.response?.data?.detail;

        setError(
          typeof detail === 'string'
            ? detail
            : error.code === 'ECONNABORTED'
              ? '업로드 요청 시간이 초과되었습니다. 서버 저장 여부를 확인하세요.'
              : error.response
                ? `영상 업로드에 실패했습니다 (HTTP ${error.response.status}).`
                : '서버에 연결할 수 없습니다.'
        );
      } else {
        setError('영상 업로드에 실패했습니다.');
      }
    } finally {
      uploadInProgress.current = false;
      setIsUploading(false);
    }
  };


  // =========================
  // 분석 요청
  // =========================

  const handleAnalysisRequest = async () => {
    if (
      !result ||
      isUploading ||
      uploadInProgress.current ||
      analysisInProgress.current ||
      analysisJob
    ) {
      return;
    }

    const version = requestVersion.current;

    analysisInProgress.current = true;
    setIsRequestingAnalysis(true);
    setAnalysisError('');

    try {
      const job = await createAnalysisJob(result.video_id);

      if (version === requestVersion.current) {
        setAnalysisJob(job);
      }
    } catch (error: unknown) {
      if (version !== requestVersion.current) return;

      if (axios.isAxiosError(error)) {
        const detail: unknown = error.response?.data?.detail;

        setAnalysisError(
          typeof detail === 'string'
            ? detail
            : error.response
              ? `분석 요청에 실패했습니다 (HTTP ${error.response.status}).`
              : '분석 API 서버에 연결할 수 없습니다.'
        );
      } else {
        setAnalysisError('분석 요청 중 오류가 발생했습니다.');
      }
    } finally {
      analysisInProgress.current = false;
      setIsRequestingAnalysis(false);
    }
  };


  // =========================
  // 분석 상태 새로고침
  // =========================

  const refreshAnalysisStatus = async () => {
    if (!analysisJob) return;

    const version = requestVersion.current;

    setAnalysisError('');

    try {
      const updatedJob = await getAnalysisJob(analysisJob.id);

      if (version === requestVersion.current) {
        setAnalysisJob(updatedJob);
      }
    } catch {
      if (version === requestVersion.current) {
        setAnalysisError('분석 상태를 불러오지 못했습니다.');
      }
    }
  };


  return (
    <section
      aria-labelledby="camera-title"
      className="flex min-w-0 flex-col overflow-hidden rounded-lg border border-line bg-surface"
    >
      {/* 상단 헤더 */}
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 bg-side px-5 py-4">
        <h2
          id="camera-title"
          className="text-sm font-bold text-white"
        >
          영상 기반 인파 모니터링
        </h2>

        <p className="flex items-center gap-4 text-xs text-side-text">
          <span className="flex items-center gap-2">
            <span
              aria-hidden="true"
              className={`h-2 w-2 rounded-full ${
                videoUrl ? 'bg-ok' : 'bg-side-text/50'
              }`}
            />
            {videoUrl ? '로컬 영상 재생 중' : '연결 대기'}
          </span>

          <span className="text-side-text/70">로컬 데모</span>
        </p>
      </div>

      {/* 영상 미리보기 */}
      <div className="monitor-stage relative flex h-[280px] min-h-0 flex-1 items-center justify-center bg-monitor sm:h-[340px] xl:min-h-[300px]">
        {videoUrl ? (
          <video
            key={videoUrl}
            src={videoUrl}
            controls
            playsInline
            className="absolute inset-0 h-full w-full object-contain"
          />
        ) : (
          <div className="relative z-10 px-6 text-center">
            <div className="monitor-reticle" aria-hidden="true">
              <span />
            </div>

            <p className="text-sm font-medium text-side-text">
              연결된 영상이 없습니다.
            </p>

            <p className="mt-1 text-xs text-side-text/70">
              동영상을 업로드하여 확인하세요.
            </p>
          </div>
        )}
      </div>

      {/* 영상 업로드 / 분석 요청 */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line bg-surface px-5 py-3">
        <p className="min-w-0 truncate text-[13px] text-ink-soft">
          {videoUrl ? (
            <>
              <span className="text-muted">선택된 영상</span>{' '}
              <span className="font-medium">{fileName}</span>
            </>
          ) : (
            <span className="text-muted">
              카메라 상태: 연결 대기
            </span>
          )}
        </p>

        <div className="flex items-center gap-2">
          <label
            className={`rounded-sm bg-brand px-3.5 py-2 text-[13px] font-semibold text-[#061526] transition-colors ${
              isUploading || isRequestingAnalysis
                ? 'cursor-not-allowed opacity-50'
                : 'cursor-pointer hover:bg-brand-strong'
            }`}
          >
            {isUploading ? '업로드 중…' : '영상 업로드'}

            <input
              type="file"
              accept={VIDEO_EXTENSIONS.join(',')}
              disabled={isUploading || isRequestingAnalysis}
              onChange={handleVideoUpload}
              className="sr-only"
            />
          </label>

          <button
            type="button"
            onClick={handleAnalysisRequest}
            disabled={
              !result ||
              isUploading ||
              isRequestingAnalysis ||
              analysisJob !== null
            }
            className="rounded-sm border border-brand px-3.5 py-2 text-[13px] font-semibold text-brand transition-colors hover:bg-brand/10 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isRequestingAnalysis
              ? '요청 중…'
              : analysisJob
                ? '분석 요청됨'
                : '분석 요청'}
          </button>
        </div>
      </div>

      {/* 업로드 정보 */}
      <div
        aria-busy={isUploading}
        className="border-t border-line bg-sunken px-5 py-3 text-xs text-ink-soft"
      >
        <p className="text-muted">
          MP4 · MOV · AVI · WEBM / 최대 200MiB
        </p>

        <p role="status" className="mt-1" aria-live="polite">
          {isUploading
            ? '영상을 서버에 업로드하고 있습니다…'
            : result
              ? '영상 업로드가 완료되었습니다.'
              : ''}
        </p>

        {error && (
          <p role="alert" className="mt-1 text-red-400">
            {error}
          </p>
        )}

        {result && (
          <dl className="mt-2 grid min-w-0 grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 break-all">
            <dt>영상 ID</dt>
            <dd>{result.video_id}</dd>

            <dt>원본 파일명</dt>
            <dd>{result.filename}</dd>

            <dt>저장 파일명</dt>
            <dd>{result.saved_filename}</dd>

            <dt>파일 크기</dt>
            <dd>{result.file_size.toLocaleString()} 바이트</dd>

            <dt>상태</dt>
            <dd>{result.status}</dd>

            <dt>데이터 출처</dt>
            <dd>{result.data_source}</dd>
          </dl>
        )}
      </div>

      {/* 분석 작업 정보 */}
      {(result || analysisError) && (
        <div className="border-t border-line bg-sunken px-5 py-3 text-xs text-ink-soft">
          <h3 className="mb-2 text-sm font-semibold">
            AI 분석 작업
          </h3>

          {analysisError && (
            <p role="alert" className="mb-2 text-red-400">
              {analysisError}
            </p>
          )}

          {analysisJob ? (
            <div className="space-y-2">
              <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 break-all">
                <dt>작업 ID</dt>
                <dd>{analysisJob.id}</dd>

                <dt>분석 상태</dt>
                <dd>
                  {STATUS_LABELS[analysisJob.status]}
                  {' '}
                  ({analysisJob.status})
                </dd>

                <dt>생성 시간</dt>
                <dd>
                  {new Date(
                    analysisJob.created_at
                  ).toLocaleString('ko-KR')}
                </dd>
              </dl>

              <button
                type="button"
                onClick={refreshAnalysisStatus}
                className="rounded-sm border border-line px-3 py-1.5 text-xs hover:bg-white/5"
              >
                상태 새로고침
              </button>

              {analysisJob.error_message && (
                <p className="text-red-400">
                  {analysisJob.error_message}
                </p>
              )}
            </div>
          ) : (
            <p className="text-muted">
              {isRequestingAnalysis
                ? '분석 작업을 생성하고 있습니다…'
                : '영상을 업로드한 다음 분석 요청을 진행할 수 있습니다.'}
            </p>
          )}
        </div>
      )}

      {/* 하단 안내 */}
      <p className="border-t border-line bg-sunken px-5 py-3 text-[11px] leading-relaxed text-muted">
        선택한 영상은 로컬 미리보기와 서버 업로드를 지원합니다.
        분석 요청 시 작업이 대기열에 등록됩니다.
        실제 YOLO 객체 탐지 및 혼잡도 분석은 AI 모델 연동 후 제공됩니다.
      </p>
    </section>
  );
}
