
import { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';

import {
  getVideos,
  getAnalysisJobsByVideo,
  createAnalysisJob,
} from '../services/api';

import type {
  VideoListItem,
  AnalysisJobResponse,
} from '../services/api';


// =====================================
// 상수 및 유틸리티
// =====================================

const POLLING_INTERVAL = 5000;

const STATUS_LABELS: Record<
  AnalysisJobResponse['status'],
  string
> = {
  queued: '분석 대기',
  running: '분석 중',
  succeeded: '분석 완료',
  failed: '분석 실패',
  cancelled: '분석 취소',
};

function isActiveJob(job: AnalysisJobResponse): boolean {
  return job.status === 'queued' || job.status === 'running';
}

function formatDate(value: string | null) {
  if (!value) return '-';

  const date = new Date(value);

  return Number.isNaN(date.getTime())
    ? '-'
    : date.toLocaleString('ko-KR');
}

function formatSize(bytes: number) {
  return `${(bytes / 1024 / 1024).toFixed(2)} MiB`;
}

function getErrorMessage(error: unknown) {
  if (axios.isAxiosError(error)) {
    const detail: unknown = error.response?.data?.detail;

    if (typeof detail === 'string') {
      return detail;
    }

    return error.response
      ? `요청 실패 (HTTP ${error.response.status})`
      : '서버에 연결할 수 없습니다.';
  }

  return '요청 처리 중 오류가 발생했습니다.';
}


// =====================================
// 영상 관리 페이지
// =====================================

export default function VideoManagement() {
  // 영상 목록
  const [videos, setVideos] = useState<VideoListItem[]>([]);

  // 선택한 영상
  const [selectedVideo, setSelectedVideo] =
    useState<VideoListItem | null>(null);

  // 분석 작업 이력
  const [jobs, setJobs] = useState<AnalysisJobResponse[]>([]);

  // 로딩 상태
  const [loading, setLoading] = useState(true);
  const [jobsLoading, setJobsLoading] = useState(false);
  const [creating, setCreating] = useState(false);

  // 오류 상태
  const [error, setError] = useState('');
  const [jobError, setJobError] = useState('');

  // 마지막 자동 갱신 시간
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  // 이전 영상의 응답이 현재 선택한 영상을 덮어쓰지 않도록 방지
  const selectionVersion = useRef(0);

  // 분석 요청 중복 실행 방지
  const creatingRef = useRef(false);

  // 진행 중인 수동 조회와 자동 조회의 중복 방지
  const jobsRequestRef = useRef(false);


  // =====================================
  // 영상 목록 조회
  // =====================================

  const fetchVideos = useCallback(() => {
    return getVideos()
      .then((response) => {
        setVideos(response);
        setError('');
      })
      .catch((err: unknown) => {
        setError(getErrorMessage(err));
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  const loadVideos = () => {
    setLoading(true);
    setError('');

    return fetchVideos();
  };

  useEffect(() => {
    void fetchVideos();
  }, [fetchVideos]);


  // =====================================
  // 영상 선택 및 분석 이력 조회
  // =====================================

  const selectVideo = async (video: VideoListItem) => {
    const version = ++selectionVersion.current;

    setSelectedVideo(video);
    setJobs([]);
    setJobError('');
    setLastUpdated(null);
    setJobsLoading(true);

    try {
      const response = await getAnalysisJobsByVideo(video.id);

      if (version === selectionVersion.current) {
        setJobs(response);
        setLastUpdated(new Date().toISOString());
      }
    } catch (err) {
      if (version === selectionVersion.current) {
        setJobError(getErrorMessage(err));
      }
    } finally {
      if (version === selectionVersion.current) {
        setJobsLoading(false);
      }
    }
  };


  // =====================================
  // 분석 요청
  // =====================================

  const requestAnalysis = async () => {
    if (
      !selectedVideo ||
      creatingRef.current ||
      jobsLoading ||
      selectedVideo.status !== 'uploaded'
    ) {
      return;
    }

    const hasActive = jobs.some(isActiveJob);

    if (hasActive) {
      setJobError(
        '이미 대기 중이거나 진행 중인 분석 작업이 있습니다.'
      );
      return;
    }

    const version = selectionVersion.current;
    const videoId = selectedVideo.id;

    creatingRef.current = true;
    setCreating(true);
    setJobError('');

    try {
      await createAnalysisJob(videoId);

      const response = await getAnalysisJobsByVideo(videoId);

      if (version === selectionVersion.current) {
        setJobs(response);
        setLastUpdated(new Date().toISOString());
      }
    } catch (err) {
      if (version === selectionVersion.current) {
        setJobError(getErrorMessage(err));
      }
    } finally {
      creatingRef.current = false;
      setCreating(false);
    }
  };


  // =====================================
  // 분석 이력 수동 새로고침
  // =====================================

  const refreshAnalysisJobs = async () => {
    if (
      !selectedVideo ||
      jobsLoading ||
      creating ||
      jobsRequestRef.current
    ) {
      return;
    }

    const version = selectionVersion.current;
    const videoId = selectedVideo.id;

    jobsRequestRef.current = true;
    setJobsLoading(true);
    setJobError('');

    try {
      const response = await getAnalysisJobsByVideo(videoId);

      if (version === selectionVersion.current) {
        setJobs(response);
        setLastUpdated(new Date().toISOString());
      }
    } catch (err) {
      if (version === selectionVersion.current) {
        setJobError(getErrorMessage(err));
      }
    } finally {
      jobsRequestRef.current = false;

      if (version === selectionVersion.current) {
        setJobsLoading(false);
      }
    }
  };


  // =====================================
  // 현재 분석 작업 상태 확인
  // =====================================

  const hasActiveJob = jobs.some(isActiveJob);


  // =====================================
  // 분석 상태 자동 갱신
  // =====================================

  useEffect(() => {
    // 선택한 영상이 없거나 활성 작업이 없으면 조회하지 않음
    if (
      !selectedVideo ||
      !hasActiveJob ||
      jobsLoading ||
      creating
    ) {
      return;
    }

    const videoId = selectedVideo.id;
    const version = selectionVersion.current;

    let cancelled = false;
    let fetching = false;

    const timer = window.setInterval(async () => {
      // 브라우저 탭이 숨겨져 있으면 요청 중단
      if (document.visibilityState === 'hidden') {
        return;
      }

      // 이전 요청이 끝나지 않았다면 중복 요청하지 않음
      if (fetching || jobsRequestRef.current) {
        return;
      }

      fetching = true;
      jobsRequestRef.current = true;

      try {
        const response = await getAnalysisJobsByVideo(videoId);

        // 선택한 영상이 변경됐거나 컴포넌트가 정리된 경우 무시
        if (
          cancelled ||
          version !== selectionVersion.current
        ) {
          return;
        }

        setJobs(response);
        setJobError('');
        setLastUpdated(new Date().toISOString());
      } catch (err) {
        if (
          !cancelled &&
          version === selectionVersion.current
        ) {
          setJobError(getErrorMessage(err));
        }
      } finally {
        fetching = false;
        jobsRequestRef.current = false;
      }
    }, POLLING_INTERVAL);

    // 영상 변경, 상태 완료, 페이지 이동 시 타이머 정리
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [
    selectedVideo?.id,
    hasActiveJob,
    jobsLoading,
    creating,
  ]);


  // =====================================
  // 영상 스트리밍 URL
  // =====================================

  const videoStreamUrl = selectedVideo
    ? `${
        (
          import.meta.env.VITE_API_BASE_URL ||
          'http://127.0.0.1:8000'
        ).replace(/\/+$/, '')
      }/api/videos/${encodeURIComponent(selectedVideo.id)}/stream`
    : null;


  // =====================================
  // 화면
  // =====================================

  return (
    <section className="space-y-5">

      {/* 페이지 제목 */}
      <div>
        <h1 className="text-xl font-bold text-ink">
          영상 관리
        </h1>

        <p className="mt-1 text-sm text-muted">
          업로드한 영상과 AI 분석 작업 이력을 관리합니다.
        </p>
      </div>


      {/* =====================================
          업로드 영상 목록
      ===================================== */}

      <div className="overflow-hidden rounded-lg border border-line bg-surface">

        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-4">
          <h2 className="text-sm font-semibold text-ink">
            업로드 영상 목록
          </h2>

          <button
            type="button"
            onClick={() => void loadVideos()}
            disabled={loading}
            className="rounded-sm border border-line px-3 py-1.5 text-xs text-ink-soft transition-colors hover:bg-sunken disabled:opacity-50"
          >
            {loading ? '불러오는 중…' : '목록 새로고침'}
          </button>
        </div>

        {error && (
          <p
            role="alert"
            className="px-5 py-3 text-sm text-red-400"
          >
            {error}
          </p>
        )}

        {loading ? (
          <p className="px-5 py-8 text-center text-sm text-muted">
            영상 목록을 불러오고 있습니다…
          </p>
        ) : videos.length === 0 ? (
          <p className="px-5 py-8 text-center text-sm text-muted">
            업로드된 영상이 없습니다.
          </p>
        ) : (
          <div className="divide-y divide-line">
            {videos.map((video) => (
              <button
                key={video.id}
                type="button"
                onClick={() => void selectVideo(video)}
                className={`flex w-full items-center justify-between gap-4 px-5 py-4 text-left transition-colors hover:bg-sunken ${
                  selectedVideo?.id === video.id
                    ? 'bg-sunken'
                    : ''
                }`}
              >
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-ink">
                    {video.filename}
                  </p>

                  <p className="mt-1 text-xs text-muted">
                    {formatDate(video.uploaded_at)}
                    {' · '}
                    {formatSize(video.file_size)}
                  </p>
                </div>

                <span className="shrink-0 rounded-sm border border-line px-2 py-1 text-xs text-ink-soft">
                  {video.status === 'uploaded'
                    ? '업로드 완료'
                    : '사용 불가'}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>


      {/* =====================================
          선택한 영상 미리보기
      ===================================== */}

      {selectedVideo && videoStreamUrl && (
        <div className="overflow-hidden rounded-lg border border-line bg-surface">

          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-4">
            <div>
              <h2 className="text-sm font-semibold text-ink">
                선택한 영상 미리보기
              </h2>

              <p className="mt-1 text-xs text-muted">
                {selectedVideo.filename}
              </p>
            </div>

            <span className="rounded-sm border border-line px-2 py-1 text-xs text-ink-soft">
              원본 영상
            </span>
          </div>

          <div className="flex items-center justify-center bg-monitor p-4">
            <video
              key={selectedVideo.id}
              src={videoStreamUrl}
              controls
              preload="metadata"
              playsInline
              className="aspect-video max-h-[440px] w-full object-contain"
            >
              브라우저에서 영상 재생을 지원하지 않습니다.
            </video>
          </div>

          <div className="border-t border-line px-5 py-3">
            <p className="text-xs text-muted">
              서버에 저장된 원본 영상입니다.
              AI 객체 탐지 결과는 모델 연동 이후 표시됩니다.
            </p>
          </div>
        </div>
      )}


      {/* =====================================
          AI 분석 작업 이력
      ===================================== */}

      {selectedVideo && (
        <div className="overflow-hidden rounded-lg border border-line bg-surface">

          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-4">

            <div className="min-w-0">
              <h2 className="truncate text-sm font-semibold text-ink">
                {selectedVideo.filename}
              </h2>

              <p className="mt-1 text-xs text-muted">
                AI 분석 작업 이력
              </p>
            </div>

            <button
              type="button"
              onClick={() => void requestAnalysis()}
              disabled={
                creating ||
                jobsLoading ||
                hasActiveJob ||
                selectedVideo.status !== 'uploaded'
              }
              className="rounded-sm bg-brand px-3 py-2 text-xs font-semibold text-[#061526] transition-colors hover:bg-brand-strong disabled:cursor-not-allowed disabled:opacity-40"
            >
              {creating
                ? '요청 중…'
                : hasActiveJob
                  ? '분석 요청됨'
                  : '분석 요청'}
            </button>
          </div>


          {/* 자동 갱신 상태 */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line bg-sunken px-5 py-3">

            <div className="flex items-center gap-2">
              <span
                aria-hidden="true"
                className={`h-2 w-2 rounded-full ${
                  hasActiveJob ? 'bg-ok' : 'bg-muted'
                }`}
              />

              <span className="text-xs text-ink-soft">
                {hasActiveJob
                  ? '분석 상태 자동 갱신 중 (5초 간격)'
                  : '자동 갱신 대기'}
              </span>
            </div>

            {lastUpdated && (
              <span className="text-xs text-muted">
                최근 확인: {formatDate(lastUpdated)}
              </span>
            )}
          </div>


          {/* 오류 메시지 */}
          {jobError && (
            <p
              role="alert"
              className="px-5 py-3 text-sm text-red-400"
            >
              {jobError}
            </p>
          )}


          {/* 작업 목록 */}
          {jobsLoading ? (
            <p className="px-5 py-6 text-sm text-muted">
              분석 이력을 불러오는 중입니다…
            </p>
          ) : jobs.length === 0 ? (
            <p className="px-5 py-6 text-sm text-muted">
              등록된 분석 작업이 없습니다.
            </p>
          ) : (
            <div className="divide-y divide-line">
              {jobs.map((job) => (
                <div
                  key={job.id}
                  className="flex flex-wrap items-center justify-between gap-3 px-5 py-4"
                >
                  <div className="min-w-0">
                    <p className="break-all text-xs text-ink-soft">
                      {job.id}
                    </p>

                    <p className="mt-1 text-xs text-muted">
                      {formatDate(job.created_at)}
                    </p>
                  </div>

                  <span className="shrink-0 rounded-sm border border-line px-2 py-1 text-xs text-ink-soft">
                    {STATUS_LABELS[job.status] ?? job.status}
                  </span>
                </div>
              ))}
            </div>
          )}


          {/* 하단 안내 및 수동 갱신 */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-5 py-3">

            <p className="text-xs text-muted">
              {hasActiveJob
                ? '분석 작업 상태를 5초마다 확인합니다. 실제 AI 분석은 모델 연동 이후 실행됩니다.'
                : '분석 상태는 데이터베이스에 저장된 작업 상태를 기준으로 표시합니다.'}
            </p>

            <button
              type="button"
              onClick={() => void refreshAnalysisJobs()}
              disabled={jobsLoading || creating}
              className="rounded-sm border border-line px-3 py-1.5 text-xs text-ink-soft hover:bg-sunken disabled:opacity-50"
            >
              분석 이력 새로고침
            </button>
          </div>

        </div>
      )}

    </section>
  );
}
