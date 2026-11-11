import {useState, useEffect} from 'react';

export default function CameraViewer() {
    const [videoUrl, setVideoUrl]  = useState<string | null>(null);
    const [fileName, setFileName] = useState<string>('');

    useEffect(() => {
        return () => {
            if (videoUrl) {
                URL.revokeObjectURL(videoUrl);
            }
        }
    }, [videoUrl]);

    const handleVideoUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];

        if(!file) return;

        if(!file.type.startsWith('video/')) {
            alert("동영상 파일만 업로드 할 수 있습니다.");
            event.target.value = '';
            return;
        }

        const url = URL.createObjectURL(file);

        setVideoUrl(url);
        setFileName(file.name);

        event.target.value = '';
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
          영상 업로드
          <input
            type="file"
            accept="video/*"
            onChange={handleVideoUpload}
            className="sr-only"
          />
        </label>
      </div>

      <p className="border-t border-line bg-sunken px-5 py-3 text-[11px] leading-relaxed text-muted">
        현재는 로컬 동영상 재생만 지원합니다.
        AI 객체 탐지 및 실시간 CCTV 연동은 추후 구현 예정입니다.
      </p>
    </section>
  );
}
