import {useMemo, useState} from 'react';
import {Brush, CartesianGrid, Legend, Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis} from 'recharts';
import type {VideoForecastModel, VideoForecastResponse, VideoForecastSource} from '../../services/api';
import {useVideoForecast} from '../../hooks/useVideoForecast';

// 시계열 예측 탭: forecast/run_job_forecast.py 결과(10·30·60초 뒤 인원 예측 + 이상 흐름 3단계)를 보여준다.
// 예측은 오프라인 산출물이라 job마다 따로 돌려야 하고, 안 돌렸으면 "결과 없음"으로 안내한다.

const SOURCE_LABEL: Record<VideoForecastSource, string> = {yolo: 'YOLO+ByteTrack 인원', dmcount: 'DM-Count 인원', fused: 'YOLO + DM-Count(보완) 인원'};
// 다크 표면(#0b2145) 기준 dataviz 검증 통과(명도 대역·CVD·대비). 실제값은 본문 잉크, 직전값은 회색 점선.
const MODEL_STYLE: Record<VideoForecastModel, {label: string; color: string; dash?: string}> = {
  persistence: {label: '직전값(기준선)', color: 'var(--color-muted)', dash: '4 4'},
  ridge_ar: {label: 'Ridge AR', color: '#1497c2'},
  lightgbm: {label: 'LightGBM(영상 내 학습)', color: '#c4861a'},
  lightgbm_pre: {label: 'LightGBM(미리 학습)', color: '#a46ff0'},
};
const MODEL_ORDER: VideoForecastModel[] = ['persistence', 'ridge_ar', 'lightgbm', 'lightgbm_pre'];
const LEVEL_FILL: Record<number, string> = {1: 'var(--color-warn)', 2: 'var(--color-danger)'};

const clock = (sec: number) => `${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, '0')}`;
const pct = (value: number | null | undefined) => value == null ? '—' : `${value > 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;

export default function VideoForecastPanel({jobId}: {jobId: string}) {
  const [source, setSource] = useState<VideoForecastSource>('yolo');
  const [horizon, setHorizon] = useState(60);
  const {state, retry} = useVideoForecast(jobId, source, horizon);

  return (
    <section aria-label="시계열 예측" className="space-y-5">
      <div className="flex flex-wrap items-center gap-4 rounded-lg border border-line bg-surface p-4 text-xs text-ink-soft">
        <Segmented label="인원 출처" value={source} options={(['yolo', 'dmcount', 'fused'] as const).map((value) => ({value, label: SOURCE_LABEL[value]}))} onChange={setSource} />
        <Segmented label="예측 시간" value={horizon} options={[10, 30, 60].map((value) => ({value, label: `${value}초 뒤`}))} onChange={setHorizon} />
      </div>
      {state.status === 'loading' ? <p role="status" className="rounded-lg border border-line bg-surface p-5 text-sm text-muted">시계열 예측 결과를 불러오는 중…</p>
        : state.status === 'not_run' ? <p role="status" className="rounded-lg border border-line bg-surface p-5 text-sm text-muted">이 작업에는 {SOURCE_LABEL[source]} 기준 시계열 예측 결과가 없습니다. <code>python -m forecast.run_job_forecast</code>로 생성할 수 있습니다.</p>
        : state.status === 'error' ? <div role="alert" className="rounded-lg border border-line bg-surface p-5 text-sm text-ink"><p>{state.message}</p><button type="button" onClick={retry} className="mt-3 rounded-sm border border-line px-3 py-2">다시 조회</button></div>
        : <ForecastView data={state.data} />}
    </section>
  );
}

function Segmented<T extends string | number>({label, value, options, onChange}: {label: string; value: T; options: {value: T; label: string}[]; onChange: (value: T) => void}) {
  return (
    <div role="radiogroup" aria-label={label} className="flex items-center gap-2">
      <span className="text-muted">{label}</span>
      <div className="flex overflow-hidden rounded-sm border border-line">
        {options.map((option) => <button key={option.value} type="button" role="radio" aria-checked={option.value === value} onClick={() => onChange(option.value)}
          className={`px-3 py-1.5 ${option.value === value ? 'bg-brand-soft font-bold text-ink' : 'text-ink-soft hover:bg-sunken'}`}>{option.label}</button>)}
      </div>
    </div>
  );
}

function ForecastView({data}: {data: VideoForecastResponse}) {
  const [zone, setZone] = useState(data.zones[0] ?? '');
  const [hidden, setHidden] = useState<Set<VideoForecastModel>>(new Set());
  const activeZone = data.zones.includes(zone) ? zone : data.zones[0] ?? '';
  const rows = useMemo(() => data.series.filter((point) => point.zone === activeZone), [data.series, activeZone]);
  const segments = useMemo(() => data.level_segments.filter((segment) => segment.zone === activeZone), [data.level_segments, activeZone]);
  const events = useMemo(() => data.level_events.filter((event) => event.zone === activeZone), [data.level_events, activeZone]);
  const backtest = data.backtest.filter((row) => row.horizon_s === data.horizon_s && MODEL_ORDER.includes(row.model as VideoForecastModel));
  const best = backtest.reduce<(typeof backtest)[number] | null>((top, row) => (top === null || row.mae < top.mae ? row : top), null);
  const levelSeconds = data.level_seconds[activeZone] ?? {};
  const toggle = (model: VideoForecastModel) => setHidden((prev) => {
    const next = new Set(prev);
    if (next.has(model)) next.delete(model); else next.add(model);
    return next;
  });

  return <>
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      <Stat label={`${data.horizon_s}초 뒤 최저 오차 모델`} value={best ? MODEL_STYLE[best.model as VideoForecastModel].label : '—'} />
      <Stat label="평균 절대 오차 (MAE)" value={best ? `${best.mae.toFixed(2)}명` : '—'} />
      <Stat label="직전값 대비 개선" value={pct(best?.skill)} tone={best && best.skill > 0 ? 'ok' : undefined} />
      <Stat label={`${activeZone} 구역 주의·높음`} value={`${levelSeconds['주의'] ?? 0}초 · ${levelSeconds['높음'] ?? 0}초`} />
    </div>

    <div className="rounded-lg border border-line bg-surface p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-sm font-bold text-ink">{activeZone} 구역 실제 인원과 {data.horizon_s}초 뒤 예측</h3>
        <label className="text-xs text-ink-soft">구역 <select aria-label="예측 구역" value={activeZone} onChange={(event) => setZone(event.target.value)} className="ml-2 rounded-sm border border-line bg-surface px-2 py-1">{data.zones.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
      </div>
      <div className="mt-3 flex flex-wrap gap-2 text-xs">
        {MODEL_ORDER.map((model) => <button key={model} type="button" aria-pressed={!hidden.has(model)} onClick={() => toggle(model)}
          className={`flex items-center gap-1.5 rounded-sm border border-line px-2 py-1 ${hidden.has(model) ? 'text-muted opacity-60' : 'text-ink-soft'}`}>
          <svg width="16" height="4" aria-hidden="true"><line x1="0" y1="2" x2="16" y2="2" stroke={MODEL_STYLE[model].color} strokeWidth="2" strokeDasharray={MODEL_STYLE[model].dash} /></svg>
          {MODEL_STYLE[model].label}</button>)}
        <span className="flex items-center gap-3 px-2 text-muted"><span className="inline-block h-3 w-3 rounded-sm bg-warn/30" /> 주의 <span className="inline-block h-3 w-3 rounded-sm bg-danger/30" /> 높음</span>
      </div>
      {!rows.length ? <p className="py-8 text-center text-sm text-muted">이 구역에는 예측 결과가 없습니다.</p> : <div aria-label="실제 인원과 예측 그래프" className="mt-4 h-80 w-full"><ResponsiveContainer width="100%" height="100%"><LineChart data={rows} margin={{top: 10, right: 20, left: 0, bottom: 5}}>
        <CartesianGrid stroke="var(--color-line)" vertical={false} />
        <XAxis dataKey="sec" type="number" domain={['dataMin', 'dataMax']} tickFormatter={clock} tick={{fill: 'var(--color-muted)', fontSize: 11}} />
        <YAxis allowDecimals unit="명" tick={{fill: 'var(--color-muted)', fontSize: 11}} width={48} />
        {segments.map((segment) => <ReferenceArea key={`${segment.start_sec}-${segment.level_num}`} x1={segment.start_sec} x2={segment.end_sec + 1} fill={LEVEL_FILL[segment.level_num]} fillOpacity={0.18} strokeOpacity={0} ifOverflow="hidden" />)}
        <Tooltip labelFormatter={(value) => `영상 ${clock(Number(value))}`} formatter={(value) => (typeof value === 'number' ? `${value.toFixed(2)}명` : '—')}
          contentStyle={{background: 'var(--color-sunken)', border: '1px solid var(--color-line)', fontSize: 12}} labelStyle={{color: 'var(--color-ink)'}} />
        <Legend wrapperStyle={{fontSize: 12}} />
        <Line dataKey="actual" name="실제(10초 평균)" type="linear" stroke="var(--color-ink)" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls={false} />
        {MODEL_ORDER.filter((model) => !hidden.has(model)).map((model) => <Line key={model} dataKey={model} name={MODEL_STYLE[model].label} type="linear" stroke={MODEL_STYLE[model].color}
          strokeWidth={model === 'persistence' ? 1.5 : 2} strokeDasharray={MODEL_STYLE[model].dash} dot={false} isAnimationActive={false} connectNulls={false} />)}
        <Brush dataKey="sec" height={22} travellerWidth={8} tickFormatter={clock} stroke="var(--color-line-strong)" fill="var(--color-sunken)" />
      </LineChart></ResponsiveContainer></div>}
      <p className="mt-2 text-xs text-muted">아래 막대를 끌어 구간을 확대할 수 있습니다. 영상 끝 이후 구간은 실제값 없이 예측만 있습니다.</p>
    </div>

    <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
      <div className="overflow-hidden rounded-lg border border-line bg-surface">
        <h3 className="px-5 py-4 text-sm font-bold text-ink">모델별 예측 오차 (전 구역, backtest)</h3>
        <div className="overflow-x-auto"><table className="w-full text-left text-xs text-ink-soft"><thead className="bg-sunken text-muted"><tr>{['예측 시간', '모델', 'MAE', 'RMSE', '직전값 대비', '비교 시점'].map((title) => <th key={title} scope="col" className="whitespace-nowrap px-4 py-3">{title}</th>)}</tr></thead>
          <tbody>{data.horizons.flatMap((h) => MODEL_ORDER.map((model) => data.backtest.find((row) => row.horizon_s === h && row.model === model)).filter((row) => row !== undefined).map((row) => {
            const bestOfHorizon = data.backtest.filter((r) => r.horizon_s === h && MODEL_ORDER.includes(r.model as VideoForecastModel)).every((r) => r.mae >= row.mae);
            return <tr key={`${h}-${row.model}`} className={`border-t border-line ${h === data.horizon_s ? '' : 'opacity-60'}`}>
              <td className="px-4 py-2.5">{h}초</td>
              <td className="whitespace-nowrap px-4 py-2.5">{MODEL_STYLE[row.model as VideoForecastModel].label}{bestOfHorizon && <span className="ml-1.5 rounded-sm bg-ok-soft px-1.5 py-0.5 text-[10px] text-ok">최저</span>}</td>
              <td className="px-4 py-2.5">{row.mae.toFixed(3)}</td>
              <td className="px-4 py-2.5">{row.rmse.toFixed(3)}</td>
              <td className={`px-4 py-2.5 ${row.skill > 0 ? 'text-ok' : ''}`}>{row.model === 'persistence' ? '기준' : pct(row.skill)}</td>
              <td className="px-4 py-2.5">{row.n.toLocaleString('ko-KR')}</td>
            </tr>;
          }))}</tbody></table></div>
        <p className="px-5 py-3 text-xs text-muted">{data.note}</p>
      </div>

      <div className="overflow-hidden rounded-lg border border-line bg-surface">
        <h3 className="px-5 py-4 text-sm font-bold text-ink">{activeZone} 구역 이상 흐름 구간 ({events.length}건)</h3>
        {!events.length ? <p className="px-5 pb-5 text-sm text-muted">주의·높음 단계 구간이 없습니다.</p> : <div className="max-h-96 overflow-auto"><table className="w-full text-left text-xs text-ink-soft"><thead className="sticky top-0 bg-sunken text-muted"><tr>{['시작', '단계', '지속', '근거'].map((title) => <th key={title} scope="col" className="whitespace-nowrap px-4 py-3">{title}</th>)}</tr></thead>
          <tbody>{events.map((event) => <tr key={`${event.start_sec}-${event.level}`} className="border-t border-line">
            <td className="whitespace-nowrap px-4 py-2.5">{clock(event.start_sec)}</td>
            <td className={`whitespace-nowrap px-4 py-2.5 font-semibold ${event.level === '높음' ? 'text-danger' : 'text-warn'}`}>{event.level === '높음' ? '▲ ' : '● '}{event.level}</td>
            <td className="whitespace-nowrap px-4 py-2.5">{event.duration_s ?? '—'}초</td>
            <td className="px-4 py-2.5">{event.reason || '—'}</td>
          </tr>)}</tbody></table></div>}
        <p className="px-5 py-3 text-xs text-muted">병목 점수·규칙 경보(급증·정체)·Isolation Forest를 합친 참고 지표입니다. 사고나 위험을 확정하지 않습니다. Isolation Forest: {data.iforest ?? '—'}</p>
      </div>
    </div>

    {data.minutes && <div className="overflow-hidden rounded-lg border border-line bg-surface">
      <h3 className="px-5 py-4 text-sm font-bold text-ink">5·10·15분 뒤 예측 (시뮬레이션 학습 모델, 참고)</h3>
      <div className="overflow-x-auto"><table className="w-full text-left text-xs text-ink-soft"><thead className="bg-sunken text-muted"><tr>{['구역 / 예측 시간', 'MAE', '직전값 MAE', '모델 적용 비율', '비교 시점'].map((title) => <th key={title} scope="col" className="whitespace-nowrap px-4 py-3">{title}</th>)}</tr></thead>
        <tbody>{Object.entries(data.minutes.backtest).map(([key, row]) => <tr key={key} className="border-t border-line">
          <td className="px-4 py-2.5">{key}</td><td className="px-4 py-2.5">{row.MAE.toFixed(3)}</td><td className="px-4 py-2.5">{row.MAE_persistence.toFixed(3)}</td>
          <td className="px-4 py-2.5">{Math.round(row.model_share * 100)}%</td><td className="px-4 py-2.5">{row.n}</td></tr>)}</tbody></table></div>
      <p className="px-5 py-3 text-xs text-muted">밀도 경보 {data.minutes.alarms}건 · 모델 {data.minutes.model_version}. 유입이 학습 범위보다 작은 시점은 직전값으로 대체되며(모델 적용 비율), 구역 면적·출구 방향은 가정값이라 밀도는 참고용입니다.</p>
    </div>}
  </>;
}

function Stat({label, value, tone}: {label: string; value: string; tone?: 'ok'}) {
  return <div className="rounded-lg border border-line bg-surface p-4"><p className="text-xs text-muted">{label}</p><p className={`mt-1 text-lg font-bold ${tone === 'ok' ? 'text-ok' : 'text-ink'}`}>{value}</p></div>;
}
