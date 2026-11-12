
import { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';
import {Link} from 'react-router-dom';
import {useAnalysisJobPolling} from '../hooks/useAnalysisJobPolling';
import VideoRoiEditor from '../components/dashboard/VideoRoiEditor';
import type {RoiReadiness} from '../components/dashboard/VideoRoiEditor';

import {
  getVideos,
  getAnalysisJobsByVideo,
  createAnalysisJob,
  getAnalysisCapabilities,
} from '../services/api';

import type {
  VideoListItem,
  AnalysisJobResponse,
} from '../services/api';


// =====================================
// 상수 및 유틸리티
// =====================================


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
  const [testEnabled,setTestEnabled]=useState<boolean|null>(null);
  useEffect(()=>{
    const controller=new AbortController();
    getAnalysisCapabilities(controller.signal).then(value=>{if(!controller.signal.aborted)setTestEnabled(value.test_analysis_enabled);}).catch(()=>{if(!controller.signal.aborted)setTestEnabled(null);});
    return ()=>controller.abort();
  },[]);
  const currentJob=jobs.find(isActiveJob)??jobs[0];

  // 로딩 상태
  const [loading, setLoading] = useState(true);
  const [jobsLoading, setJobsLoading] = useState(false);
  const [creating, setCreating] = useState(false);

  // 오류 상태
  const [error, setError] = useState('');
  const [jobError, setJobError] = useState('');
  const [detectionChoice,setDetectionChoice]=useState('');
  const detectionJobs=jobs.filter(job=>job.status==='succeeded');
  const selectedDetectionJob=jobs.find(job=>job.id===detectionChoice);
  const detectionJob=selectedDetectionJob?(selectedDetectionJob.status==='succeeded'?selectedDetectionJob:undefined):detectionJobs[0];
  const [roiReady, setRoiReady] = useState<RoiReadiness | null>(null);
  const [analysisMode, setAnalysisMode] = useState<'production' | 'test'>('production');
  const [analysisProfile, setAnalysisProfile] = useState<'fast' | 'full'>('fast');
  const onRoiReadiness = useCallback((state: RoiReadiness) => setRoiReady(state), []);
  const canAnalyze = roiReady?.videoId === selectedVideo?.id && !!roiReady?.[analysisMode] && (analysisMode!=='test'||testEnabled===true);
  const analysisMessage=jobsLoading?'분석 작업 조회 중…':jobError&&!jobs.length?'분석 작업 조회에 실패했습니다.':!currentJob?'아직 분석하지 않은 영상입니다':currentJob.status==='queued'?'분석 대기 중':currentJob.status==='running'?'AI 분석 중':currentJob.status==='failed'?`분석 실패: ${currentJob.error_message??'상세 오류 없음'}`:currentJob.status==='succeeded'?'분석 완료':'분석 취소';

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
      !canAnalyze ||
      creatingRef.current ||
      (analysisMode === 'test' && testEnabled !== true) ||
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
      const created=await createAnalysisJob(videoId, analysisMode, analysisProfile);
      if(version===selectionVersion.current){
        setJobs(previous=>[created,...previous.filter(job=>job.id!==created.id)]);
        setDetectionChoice(created.id);
      }

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

  useAnalysisJobPolling(selectedVideo?.id, hasActiveJob && !jobsLoading && !creating,
    (response) => {setJobs(response); setJobError(''); setLastUpdated(new Date().toISOString());},
    (err) => setJobError(getErrorMessage(err)), jobsRequestRef, selectionVersion);


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

      {selectedVideo&&<section aria-label="analysis-workflow" className="space-y-3 rounded-lg border border-line bg-surface p-5">
        <h2 className="text-sm font-semibold text-ink">{analysisMessage}</h2>
        <p className="break-all text-xs text-muted">{selectedVideo.filename} · video_id: {selectedVideo.id}</p>
        {currentJob&&<p className="break-all text-xs text-muted">job_id: {currentJob.id} · {currentJob.analysis_mode} · {currentJob.analysis_profile==='fast'?'FAST 빠른 분석':'FULL 정밀 분석'} {currentJob.status==='succeeded'?'완료':''}</p>}
        <p className="text-xs text-muted">분석 요청은 대기열에 작업을 등록합니다. 별도 Worker가 실행되어야 시작하며 상태는 5초마다 확인합니다. FAST는 탐지·인원·유입/유출·개인 방향을 계산합니다. FULL은 Optical Flow·병목 분석을 추가하므로 더 오래 걸립니다. 처리 시간은 장치와 영상에 따라 달라집니다.</p>
        <p className="text-xs text-muted">테스트 분석: {testEnabled===null?'서버 허용 여부 확인 안 됨':testEnabled?'API 서버에서 허용됨 · Worker도 별도 테스트 허용 필요':'현재 API 서버에서 비활성화됨'}</p>
        {currentJob?.roi_configured===false&&isActiveJob(currentJob)&&<p role="alert" className="text-xs text-warn">이 작업은 요청 당시 유효한 ROI가 없었습니다. 지금 ROI를 저장해도 기존 작업의 스냅샷은 바뀌지 않아 Worker 실행 시 실패합니다. 기존 작업 정리와 새 요청은 별도 조치가 필요하며 이 화면에서 상태를 임의 변경하지 않습니다.</p>}
        <div className="flex flex-wrap items-center gap-3"><label className="text-xs text-ink-soft">분석 모드 <select aria-label="분석 모드" value={analysisMode} onChange={e=>setAnalysisMode(e.target.value as 'production' | 'test')} disabled={creating} className="rounded-sm border border-line bg-surface p-2"><option value="production">운영 — 승인 ROI 필요</option><option value="test" disabled={testEnabled!==true}>테스트 — 서버 허용 필요</option></select></label>
            <label className="text-xs text-ink-soft">분석 범위 <select aria-label="분석 범위" value={hasActiveJob?(currentJob?.analysis_profile??'full'):analysisProfile} onChange={e=>setAnalysisProfile(e.target.value as 'fast'|'full')} disabled={creating||hasActiveJob} className="rounded-sm border border-line bg-surface p-2"><option value="fast">FAST — 빠른 분석</option><option value="full">FULL — 정밀 분석</option></select></label>
            <p className="text-xs text-muted">{hasActiveJob?'현재 작업의 분석 범위를 표시합니다. 대기·분석 중에는 변경할 수 없으며 완료 후 FAST 또는 FULL을 선택해 새 작업을 요청할 수 있습니다.':'FAST에서는 Optical Flow와 병목 위험도를 분석하지 않습니다. FULL은 두 분석을 추가로 수행합니다.'}</p>
            <button
              type="button"
              onClick={() => void requestAnalysis()}
              disabled={
                creating ||
                jobsLoading ||
                hasActiveJob ||
                !canAnalyze ||
                selectedVideo.status !== 'uploaded'
              }
              className="rounded-sm bg-brand px-3 py-2 text-xs font-semibold text-[#061526] transition-colors hover:bg-brand-strong disabled:cursor-not-allowed disabled:opacity-40"
            >
              {creating
                ? '요청 중…'
                : hasActiveJob
                  ? '분석 요청됨'
                  : '분석 요청'}
            </button></div>
        {!canAnalyze&&<p className="text-xs text-warn">미저장 ROI는 먼저 미리보기 후 후보로 저장하세요. 운영 분석에는 승인된 ROI가 필요합니다. 후보 테스트는 API 서버와 Worker 모두 테스트 허용이 필요합니다. 기존 ROI 수정 초안은 현재 서버에 새 버전으로 저장할 수 없습니다.</p>}
      </section>}

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

          <label className="block px-5 py-3 text-xs text-ink-soft">탐지 결과 작업 <select aria-label="탐지 결과 작업" value={detectionJob?.id??''} onChange={e=>setDetectionChoice(e.target.value)} className="mt-2 block w-full rounded-sm border border-line bg-surface p-2"><option value="" disabled>완료된 작업 없음</option>{detectionJobs.map(job=><option key={job.id} value={job.id}>{job.analysis_profile==='fast'?'FAST':'FULL'} · {job.analysis_mode==='test'?'테스트':'운영'} · {formatDate(job.finished_at)} · {job.id}</option>)}</select></label>
          <VideoRoiEditor key={selectedVideo.id} videoId={selectedVideo.id} onReadiness={onRoiReadiness} detectionJobId={detectionJob?.id??null} analysisMessage={analysisMessage}/>

          <div className="border-t border-line px-5 py-3">
            <p className="text-xs text-muted">
              서버에 저장된 원본 영상입니다.
                분석 수치는 작업별 대시보드 결과에서 확인할 수 있습니다.
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

            
          </div>


          {/* 자동 갱신 상태 */}
          {!canAnalyze && <p className="px-5 py-3 text-xs text-warn">운영 분석은 저장된 승인 ROI가 필요합니다. 테스트 분석은 저장된 후보 ROI도 사용할 수 있지만 API 서버와 Worker의 테스트 허용 설정이 필요합니다. 미저장 변경 또는 미완성 폴리곤이 있으면 분석을 요청할 수 없습니다.</p>}
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
                    <p className="mt-1 text-xs text-muted">{job.analysis_mode === 'test' ? '테스트 분석' : '운영 분석'}</p>
                    {job.status === 'failed' && <p role="alert" className="mt-2 text-xs text-danger">{job.error_message || '분석 실패: 상세 메시지가 없습니다.'}</p>}
                    {job.status === 'succeeded' && <Link className="mt-2 inline-block text-xs text-brand underline" to={`/dashboard?job=${encodeURIComponent(job.id)}`}>대시보드에서 실제 결과 보기</Link>}
                  </div>

                  <span className="shrink-0 rounded-sm border border-line px-2 py-1 text-xs text-ink-soft">
                    {job.analysis_profile==='fast'?'FAST':'FULL'} · {STATUS_LABELS[job.status] ?? job.status}
                  </span>
                </div>
              ))}
            </div>
          )}


          {/* 하단 안내 및 수동 갱신 */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-5 py-3">

            <p className="text-xs text-muted">
              {hasActiveJob
                ? '분석 작업 상태를 5초마다 확인합니다. 별도 Worker가 실제 분석을 처리합니다.'
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
