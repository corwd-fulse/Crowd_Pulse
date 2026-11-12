// Keep aligned with backend/routers/videos.py.
export const VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.webm'];
export const MAX_VIDEO_SIZE = 200 * 1024 * 1024;

export function validateVideo(file: Pick<File, 'name' | 'size'>): string | null {
  const extension = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
  if (!VIDEO_EXTENSIONS.includes(extension)) {
    return 'MP4, MOV, AVI, WEBM 형식의 영상만 업로드할 수 있습니다.';
  }
  if (file.size > MAX_VIDEO_SIZE) {
    return '영상 크기는 최대 200MiB (209,715,200바이트)까지 허용됩니다.';
  }
  return null;
}
