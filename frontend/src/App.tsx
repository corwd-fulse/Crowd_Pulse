import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

import Header from './components/layout/Header';
import SideBar from './components/layout/Sidebar';
import Dashboard from './pages/DashBoard';
import ComingSoon from './pages/CommingSoon';


function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-canvas lg:flex">
        <SideBar />

        <main className="min-w-0 flex-1 px-4 py-5 md:px-6 lg:px-8 lg:py-6">
          <Header />

          <div className="mt-6">

            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />

              <Route
                path="/monitoring"
                element={
                  <ComingSoon
                    title="실시간 모니터링"
                    description="CCTV 영상과 AI 객체 탐지 결과를 확인하는 페이지입니다."
                  />
                }
              />

              <Route
                path="/flow"
                element={
                  <ComingSoon
                    title="인파 흐름 분석"
                    description="인파의 이동 방향과 병목 현상을 분석하는 페이지입니다."
                  />
                }
              />

              <Route
                path="/prediction"
                element={
                  <ComingSoon
                    title="혼잡도 예측"
                    description="시계열 AI를 활용한 혼잡도 예측 결과를 확인하는 페이지입니다."
                  />
                }
              />

              <Route
                path="/alerts"
                element={
                  <ComingSoon
                    title="위험 이벤트"
                    description="인파 밀집 및 위험 이벤트 발생 기록을 관리하는 페이지입니다."
                  />
                }
              />

              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>

          </div>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
