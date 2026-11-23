import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    watch: {
      // Windows 호스트의 Docker 바인드 마운트는 파일 변경 이벤트(inotify)가 전달되지 않아 폴링으로 감지한다.
      usePolling: true,
    },
  },
})