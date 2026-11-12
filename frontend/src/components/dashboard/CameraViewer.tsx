
import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import MonitoringPlayer from './MonitoringPlayer';

import {
  uploadVideo,
  createAnalysisJob,
  getAnalysisJob,
  getVideoRoi,
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


export default function CameraViewer({onJobCreated, trackedJob}: {onJobCreated?: (job: AnalysisJobResponse) => void; trackedJob?: AnalysisJobResponse | null}) {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState('');
  const [localForJob,setLocalForJob]=useState<string|null>();
  const showLocal=!!videoUrl&&localForJob===(trackedJob?.id??null);
  const showServer=!!trackedJob&&!showLocal;

  const [isUploading, setIsUploading] = useState(false);
  const [isRequestingAnalysis, setIsRequestingAnalysis] = useState(false);
  const [analysisProfile,setAnalysisProfile]=useState<'fast'|'full'>('fast');

  const [error, setError] = useState('');
  const [analysisError, setAnalysisError] = useState('');

  const [result, setResult] = useState<VideoUploadResponse | null>(null);
  const [requestedJob, setAnalysisJob] = useState<AnalysisJobResponse | null>(null);
  const analysisJob = requestedJob && trackedJob?.id === requestedJob.id && (requestedJob.status === 'queued' || requestedJob.status === 'running') ? trackedJob : requestedJob;

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
    setLocalForJob(trackedJob?.id??null);
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
      const roi=await getVideoRoi(result.video_id);
      if(!roi||roi.roi_snapshot.status!=='approved'||roi.roi_snapshot.approved_for_analysis!==true){
        if(version===requestVersion.current)setAnalysisError('분석 요청 전 영상 관리에서 ROI를 저장하고 승인을 확인하세요. 후보 ROI 테스트는 영상 관리의 테스트 모드를 사용하세요.');
        return;
      }
      const job = await createAnalysisJob(result.video_id, 'production', analysisProfile);

      if (version === requestVersion.current) {
        setAnalysisJob(job);
        onJobCreated?.(job);
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
                showServer||videoUrl ? 'bg-ok' : 'bg-side-text/50'
              }`}
            />
            {showServer?'서버 영상 재생':showLocal?'로컬 영상 미리보기':'연결 대기'}
          </span>

          <span className="text-side-text/70">{showServer?'선택한 분석 작업 연결':'탐지 결과 연결 안 됨'}</span>
        </p>
      </div>

      {/* 영상 미리보기 */}
      <div className="monitor-stage relative min-w-0 bg-monitor" style={{height:'auto',minHeight:0,flexShrink:0}}>
        {showServer&&trackedJob?<MonitoringPlayer key={trackedJob.id} job={trackedJob}/>:showLocal&&videoUrl ? (
          <video
            key={videoUrl}
            src={videoUrl}
            controls
            playsInline
            aria-label="대시보드 로컬 영상"
            className="block max-h-[440px] w-full object-contain"
          />
        ) : (
          <div className="relative z-10 px-6 py-16 text-center">
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

      <div className="flex flex-wrap gap-2 border-t border-line px-5 py-3 text-xs text-ink-soft">
        {trackedJob&&<button type="button" disabled={showServer} onClick={()=>{setLocalForJob(undefined);}} className="rounded-sm border border-line px-3 py-2 disabled:opacity-40">선택 작업의 서버 영상 보기</button>}
        {videoUrl&&<button type="button" disabled={showLocal} onClick={()=>{setLocalForJob(trackedJob?.id??null);}} className="rounded-sm border border-line px-3 py-2 disabled:opacity-40">업로드 영상 미리보기</button>}
        {showLocal&&<p>{isUploading?'로컬 파일 재생 · 서버 업로드 중':result?'로컬 파일 재생 · 서버 업로드 완료 · 선택 작업의 탐지 결과는 겹치지 않습니다.':'로컬 파일만 재생 · 서버 업로드 미완료 · 탐지 결과 없음'}</p>}
      </div>

      {/* 영상 업로드 / 분석 요청 */}
      <label className="block px-4 py-2 text-xs text-ink-soft">분석 범위 <select aria-label="대시보드 분석 범위" value={analysisProfile} onChange={e=>setAnalysisProfile(e.target.value as 'fast'|'full')} disabled={isRequestingAnalysis||!!analysisJob} className="rounded-sm border border-line bg-surface p-2"><option value="fast">FAST — 빠른 분석</option><option value="full">FULL — 정밀 분석</option></select></label>
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line bg-surface px-5 py-3">
        <p className="min-w-0 truncate text-[13px] text-ink-soft">
          {showServer&&trackedJob?<span className="break-all">선택 작업의 서버 영상 · {trackedJob.video_id}</span>:videoUrl ? (
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
          <a href="/videos" className="mb-2 inline-block text-brand underline">영상 관리에서 ROI와 분석 작업 확인</a>

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
        실제 분석은 별도 Worker에서 처리하며, 완료된 작업의 저장 결과를 대시보드에서 확인할 수 있습니다.
      </p>
    </section>
  );
}
