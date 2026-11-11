
import { useEffect, useState } from 'react';

import StartCard from '../components/dashboard/StartCard';
import CameraViewer from '../components/dashboard/CameraViewer';
import RiskPanel from '../components/dashboard/RiskPanel';
import CrowdChart from '../components/dashboard/CrowdChart';
import FacilityMap from '../components/dashboard/FacilityMap';
import ServerStatus from '../components/common/ServiceStatus';

import {
  getDashboardSummary,
  type DashboardSummary,
} from '../services/api';

export default function Dashboard() {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadSummary() {
      try {
        const data = await getDashboardSummary();

        if (active) {
          setSummary(data);
          setError(null);
        }
      } catch {
        if (active) {
          setError('대시보드 데이터를 불러오지 못했습니다.');
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    void loadSummary();

    const interverId = window.setInterval(() => {
        void loadSummary();
    }, 5000);

    return () => {
      active = false;
      window.clearInterval(interverId);
    };
  }, []);

  // API 값이 없을 때 샘플 수치를 실제 값처럼 표시하지 않음
  const personCount = loading
    ? '불러오는 중'
    : summary
      ? `${summary.person_count.toLocaleString()}명`
      : '—';

  const congestion = loading
    ? '불러오는 중'
    : summary
      ? `${summary.congestion_percent}%`
      : '—';

  const bottleneckCount = loading
    ? '불러오는 중'
    : summary
      ? `${summary.bottleneck_count}곳`
      : '—';

  const alertCount = loading
    ? '불러오는 중'
    : summary
      ? `${summary.alert_count}건`
      : '—';

  return (
    <div className="command-dashboard space-y-5">
      <ServerStatus />

      {/* API 오류 */}
      {error && (
        <div
          role="alert"
          className="border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-600"
        >
          {error}
        </div>
      )}

      {/* 핵심 지표 */}
      <section aria-labelledby="summary-title">
        <div className="mb-2 flex items-baseline justify-between">
          <h2
            id="summary-title"
            className="text-sm font-bold text-ink"
          >
            실시간 인파 현황
          </h2>

          <span className="text-xs text-muted">
            {loading
              ? '데이터 확인 중'
              : summary?.data_source === 'mock'
                ? '샘플 데이터'
                : summary
                  ? `데이터 출처: ${summary.data_source}`
                  : '데이터 없음'}
          </span>
        </div>

        <div className="summary-grid grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <div className="min-w-0">
            <StartCard
              featured
              title="현재 탐지 인원"
              value={personCount}
              description="현재 카메라 분석 기준"
            />
          </div>

          <div className="min-w-0">
            <StartCard
              title="전체 혼잡도"
              value={congestion}
              description="샘플 혼잡도 지표"
              meter={summary?.congestion_percent ?? 0}
            />
          </div>

          <div className="min-w-0">
            <StartCard
              title="병목 의심 구역"
              value={bottleneckCount}
              description="이동 흐름 분석 기준"
              color="amber"
            />
          </div>

          <div className="min-w-0">
            <StartCard
              title="위험 알림"
              value={alertCount}
              description="확인 필요한 이벤트"
              color="rose"
            />
          </div>
        </div>
      </section>

      {/* 기존 시설 지도 유지 */}
      <FacilityMap />

      {/* 기존 CCTV 및 위험 구역 화면 유지 */}
      <section className="grid grid-cols-1 items-stretch gap-5 xl:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]">
        <CameraViewer />
        <RiskPanel />
      </section>

      {/* 기존 그래프 유지 */}
      <section>
        <CrowdChart />
      </section>
    </div>
  );
}
