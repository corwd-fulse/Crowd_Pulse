import {ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ReferenceLine} from "recharts";

const crowdData = [
  { time: '09:00', current: 80, predicted: 85 },
  { time: '09:30', current: 120, predicted: 125 },
  { time: '10:00', current: 155, predicted: 150 },
  { time: '10:30', current: 190, predicted: 205 },
  { time: '11:00', current: 225, predicted: 230 },
  { time: '11:30', current: 248, predicted: 260 },
  { time: '12:00', current: null, predicted: 285 },
  { time: '12:30', current: null, predicted: 310 },
];

// 요약 수치는 위 데이터에서 계산한다 (실제 연동 시에도 그대로 동작)
const measured = crowdData.filter((d) => d.current !== null);
const latest = measured[measured.length - 1];
const last = crowdData[crowdData.length - 1];

export default function CrowdChart() {
    return (
    <div className="rounded-lg border border-line bg-surface p-5">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-x-6 gap-y-2">
        <div>
          <h2 className="text-sm font-bold text-ink">
            시간대별 인파 변화
          </h2>
          <p className="mt-0.5 text-xs text-muted">
            현재 탐지 인원과 예상 인원 추이 (샘플 데이터)
          </p>
        </div>

        <p className="num text-[13px] text-ink-soft">
          <span className="text-muted">{latest.time} 현재 </span>
          <span className="font-bold text-ink">{latest.current}명</span>
          <span className="ml-5 text-muted">{last.time} 예측 </span>
          <span className="font-bold text-ink">{last.predicted}명</span>
        </p>
      </div>

      <div className="h-56 w-full lg:h-60">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={crowdData}
            margin={{ top: 10, right: 12, left: -12, bottom: 0 }}
          >
            <CartesianGrid
              stroke="var(--color-line)"
              vertical={false}
            />

            <XAxis
              dataKey="time"
              stroke="var(--color-line-strong)"
              tick={{ fontSize: 12, fill: 'var(--color-muted)' }}
              tickLine={false}
              padding={{ left: 12, right: 12 }}
              minTickGap={28}
            />

            <YAxis
              stroke="var(--color-line-strong)"
              tick={{ fontSize: 12, fill: 'var(--color-muted)' }}
              tickLine={false}
              axisLine={false}
              allowDecimals={false}
              unit="명"
            />

            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--color-surface)',
                border: '1px solid var(--color-line-strong)',
                borderRadius: 0,
                boxShadow: 'none',
                fontSize: 13,
              }}
              labelStyle={{ color: 'var(--color-muted)', fontWeight: 600 }}
              itemStyle={{ color: 'var(--color-ink)' }}
            />

            <Legend
              verticalAlign="top"
              align="right"
              height={28}
              iconType="plainline"
              wrapperStyle={{ fontSize: 12, color: 'var(--color-ink-soft)' }}
            />

            <ReferenceLine
              x={latest.time}
              stroke="var(--color-ink-soft)"
              strokeDasharray="2 3"
              label={{
                value: '현재',
                position: 'insideTopLeft',
                fontSize: 11,
                fill: 'var(--color-ink-soft)',
              }}
            />

            <Line
              type="monotone"
              dataKey="current"
              name="탐지 인원"
              stroke="var(--color-brand)"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 4 }}
              connectNulls={false}
            />

            <Line
              type="monotone"
              dataKey="predicted"
              name="예측 인원"
              stroke="var(--color-warn)"
              strokeWidth={1.75}
              strokeDasharray="5 4"
              dot={false}
              activeDot={{ r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p className="mt-2 text-xs text-muted">
        그래프 수치는 시연용 샘플 데이터이며 실제 AI 분석 결과가 아닙니다.
      </p>
    </div>
  );
}
