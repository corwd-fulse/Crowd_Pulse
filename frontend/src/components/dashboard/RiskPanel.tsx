import {useState} from "react";
import {ChevronDown} from "lucide-react";

type RiskLevel = 'danger' | 'warning' | 'normal';

interface RiskZone {
    id: string
    name: string
    location: string
    personCount: number
    congestion: number
    level: RiskLevel
}

const riskZones: RiskZone[] = [
  {
    id: 'zone-a',
    name: 'A구역 - 중앙 출입구',
    location: 'CAM 01',
    personCount: 125,
    congestion: 88,
    level: 'danger',
  },
  {
    id: 'zone-b',
    name: 'B구역 - 메인 통로',
    location: 'CAM 02',
    personCount: 78,
    congestion: 65,
    level: 'warning',
  },
  {
    id: 'zone-c',
    name: 'C구역 - 야외 광장',
    location: 'CAM 03',
    personCount: 45,
    congestion: 32,
    level: 'normal',
  },
];

// 상태색은 이 목록에서만 쓴다. 위험 행만 왼쪽 선으로 한 번 더 강조.
const riskStyles = {
  danger: {
    label: '위험',
    text: 'text-danger',
    dot: 'bg-danger',
    bar: 'bg-danger',
    rail: 'border-l-danger',
  },
  warning: {
    label: '주의',
    text: 'text-warn',
    dot: 'bg-warn',
    bar: 'bg-warn',
    rail: 'border-l-transparent',
  },
  normal: {
    label: '정상',
    text: 'text-ok',
    dot: 'bg-ok',
    bar: 'bg-ok',
    rail: 'border-l-transparent',
  },
};

export default function RiskPanel() {
    const [expandedId, setExpandedId] = useState<string | null>(null);

    const count = (level: RiskLevel) =>
      riskZones.filter((zone) => zone.level === level).length;

  return (
    <section
      aria-labelledby="risk-title"
      className="flex min-w-0 flex-col overflow-hidden rounded-lg border border-line bg-surface"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 border-b border-line px-4 py-3">
        <h2 id="risk-title" className="text-sm font-bold text-ink">
          구역별 위험 현황
        </h2>
        <p className="num flex gap-3 text-xs text-muted">
          <span>위험 <b className="font-semibold text-danger">{count('danger')}</b></span>
          <span>주의 <b className="font-semibold text-warn">{count('warning')}</b></span>
          <span>정상 <b className="font-semibold text-ok">{count('normal')}</b></span>
        </p>
      </div>

      <ul className="flex flex-1 flex-col">
        {riskZones.map((zone) => {
          const style = riskStyles[zone.level];
          const isExpanded = expandedId === zone.id;
          const detailId = `${zone.id}-detail`;

          return (
            <li
              key={zone.id}
              className={`flex flex-1 flex-col justify-center border-b border-l-2 border-b-line last:border-b-0 ${style.rail}`}
            >
              <button
                type="button"
                aria-expanded={isExpanded}
                aria-controls={detailId}
                onClick={() =>
                  setExpandedId(isExpanded ? null : zone.id)
                }
                className="w-full px-4 py-3.5 text-left transition-colors hover:bg-sunken"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-semibold text-ink">
                      {zone.name}
                    </p>
                    <p className="num mt-0.5 text-xs text-muted">
                      {zone.location}
                    </p>
                  </div>

                  <div className="flex shrink-0 items-center gap-2.5">
                    <span
                      className={`flex items-center gap-1.5 text-[13px] font-semibold ${style.text}`}
                    >
                      <span
                        aria-hidden="true"
                        className={`h-2 w-2 rounded-full ${style.dot}`}
                      />
                      {style.label}
                    </span>
                    <ChevronDown
                      size={15}
                      aria-hidden="true"
                      className={`text-muted transition-transform ${
                        isExpanded ? 'rotate-180' : ''
                      }`}
                    />
                  </div>
                </div>

                <div className="mt-3 flex items-center gap-3">
                  <div
                    role="meter"
                    aria-label={`${zone.name} 혼잡 지표`}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={zone.congestion}
                    className="h-1.5 flex-1 bg-line"
                  >
                    <div
                      className={`h-full ${style.bar}`}
                      style={{ width: `${zone.congestion}%` }}
                    />
                  </div>
                  <span className="num w-20 shrink-0 text-right text-xs text-ink-soft">
                    <span className="text-muted">혼잡 지표 </span>
                    <span className="font-semibold">{zone.congestion}%</span>
                  </span>
                </div>
              </button>

              {isExpanded && (
                <dl
                  id={detailId}
                  className="grid grid-cols-2 gap-4 border-t border-line bg-sunken px-4 py-3"
                >
                  <div>
                    <dt className="text-xs text-muted">탐지 인원</dt>
                    <dd className="num mt-0.5 text-sm font-semibold text-ink">
                      {zone.personCount}명
                    </dd>
                  </div>
                  <div>
                    <dt className="text-xs text-muted">위험 단계</dt>
                    <dd className={`mt-0.5 text-sm font-semibold ${style.text}`}>
                      {style.label}
                    </dd>
                  </div>
                </dl>
              )}
            </li>
          );
        })}
      </ul>

      <p className="border-t border-line bg-sunken px-4 py-3 text-[11px] leading-relaxed text-muted">
        혼잡 지표와 위험 단계는 시연용 샘플 데이터이며 실제 운영 기준이 아닙니다.
      </p>
    </section>
  );
}
