import {useState, useEffect, useRef} from 'react';
import axios from 'axios';
import {uploadVideo} from '../../services/api';
import type {VideoUploadResponse} from '../../services/api';
import {validateVideo, VIDEO_EXTENSIONS} from '../../services/videoUpload';

export default function CameraViewer() {
    const [videoUrl, setVideoUrl]  = useState<string | null>(null);
    const [fileName, setFileName] = useState<string>('');
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState('');
    const [result, setResult] = useState<VideoUploadResponse | null>(null);
    const uploadInProgress = useRef(false);

    useEffect(() => {
        return () => {
            if (videoUrl) {
                URL.revokeObjectURL(videoUrl);
            }
        }
    }, [videoUrl]);

    const handleVideoUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        event.target.value = '';

        if(!file || uploadInProgress.current) return;

        setResult(null);
        const validationError = validateVideo(file);
        setError(validationError ?? '');
        if (validationError) {
            return;
        }

        const url = URL.createObjectURL(file);

        setVideoUrl(url);
        setFileName(file.name);

        uploadInProgress.current = true;
        setIsUploading(true);
        try {
            setResult(await uploadVideo(file));
        } catch (error: unknown) {
            if (axios.isAxiosError(error)) {
                const detail: unknown = error.response?.data?.detail;
                setError(typeof detail === 'string' ? detail :
                    error.code === 'ECONNABORTED'
                        ? '업로드 요청 시간이 초과되었습니다. 서버 저장 여부를 확인한 후 다시 시도하세요.'
                        : error.response
                            ? `영상 업로드에 실패했습니다 (HTTP ${error.response.status}). 다시 시도하세요.`
                            : '서버에 연결할 수 없습니다. 백엔드 실행 상태와 네트워크를 확인하세요.');
            } else {
                setError('영상 업로드에 실패했습니다. 다시 시도하세요.');
            }
        } finally {
            uploadInProgress.current = false;
            setIsUploading(false);
        }
    };

    return (
    <section aria-labelledby="camera-title" className="flex min-w-0 flex-col overflow-hidden rounded-lg border border-line bg-surface">
      {/* 모니터 상단 정보 줄 */}
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 bg-side px-5 py-4">
        <h2 id="camera-title" className="text-sm font-bold text-white">
          실시간 인파 모니터링
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

      {/* 영상 화면 */}
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
            <div className="monitor-reticle" aria-hidden="true"><span /></div>
            <p className="text-sm font-medium text-side-text">
              연결된 영상이 없습니다.
            </p>
            <p className="mt-1 text-xs text-side-text/70">
              동영상을 업로드하여 확인하세요.
            </p>
          </div>
        )}
      </div>

      {/* 하단 제어 줄 */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line bg-surface px-5 py-3">
        <p className="min-w-0 truncate text-[13px] text-ink-soft">
          {videoUrl ? (
            <>
              <span className="text-muted">선택된 영상</span>{' '}
              <span className="font-medium">{fileName}</span>
            </>
          ) : (
            <span className="text-muted">카메라 상태: 연결 대기</span>
          )}
        </p>

        <label className="cursor-pointer rounded-sm bg-brand px-3.5 py-2 text-[13px] font-semibold text-[#061526] transition-colors hover:bg-brand-strong focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-brand">
          {isUploading ? '업로드 중…' : '영상 업로드'}
          <input
            type="file"
            accept={VIDEO_EXTENSIONS.join(',')}
            disabled={isUploading}
            onChange={handleVideoUpload}
            className="sr-only"
          />
        </label>
      </div>

      <div aria-busy={isUploading} className="border-t border-line bg-sunken px-5 py-3 text-xs text-ink-soft">
        <p className="text-muted">MP4 · MOV · AVI · WEBM / 최대 200MiB</p>
        <p role="status" className="mt-1" aria-live="polite">
          {isUploading ? '영상을 서버에 업로드하고 있습니다…' : result ? '영상 업로드가 완료되었습니다.' : ''}
        </p>
        {error && <p role="alert" className="mt-1">{error}</p>}
        {result && (
          <dl className="mt-2 grid min-w-0 grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 break-all">
            <dt>영상 ID</dt><dd>{result.video_id}</dd>
            <dt>원본 파일명</dt><dd>{result.filename}</dd>
            <dt>저장 파일명</dt><dd>{result.saved_filename}</dd>
            <dt>파일 크기</dt><dd>{result.file_size.toLocaleString()} 바이트</dd>
            <dt>상태</dt><dd>{result.status}</dd>
            <dt>데이터 출처</dt><dd>{result.data_source}</dd>
          </dl>
        )}
      </div>

      <p className="border-t border-line bg-sunken px-5 py-3 text-[11px] leading-relaxed text-muted">
        선택한 영상은 로컬 미리보기와 서버 업로드를 지원합니다. 브라우저에 따라 일부 영상 형식은 미리보기가 제한될 수 있습니다.
        AI 객체 탐지 및 실시간 CCTV 연동은 추후 구현 예정입니다.
      </p>
    </section>
  );
}
