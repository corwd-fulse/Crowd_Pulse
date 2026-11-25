import {useEffect, useMemo, useRef, useState} from 'react';
import {CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis} from 'recharts';
import {api} from '../../services/api';
import {useModelComparison} from '../../hooks/useModelComparison';

// YOLO/DM-Count 비교 탭. 두 모델은 서로 완전히 독립된 파이프라인이고 DM-Count는 정확도
// 미검증 참고 지표라서(backend density_estimation.py), 병목 점수·위험 등급과는 무관하게
// 여기서만 비교 표시한다. /results/model-comparison 응답 하나로 그래프·표·영상·히트맵을 전부 그린다.
export default function ModelComparisonPanel({jobId, videoId}: {jobId: string; videoId: string}) {
  const {state, retry} = useModelComparison(jobId);

  if (state.status === 'idle' || state.status === 'loading') {
    return <p role="status" className="rounded-lg border border-line bg-surface p-5 text-sm text-muted">모델 비교 결과를 불러오는 중…</p>;
  }
  if (state.status === 'not_run') {
    return <p role="status" className="rounded-lg border border-line bg-surface p-5 text-sm text-muted">이 작업에서는 DM-Count(보조 밀도 추정)를 실행하지 않았습니다. 비교할 결과가 없습니다.</p>;
  }
  if (state.status === 'error') {
    return <div role="alert" className="rounded-lg border border-line bg-surface p-5 text-sm text-ink"><p>{state.message}</p><button type="button" onClick={retry} className="mt-3 rounded-sm border border-line px-3 py-2">다시 조회</button></div>;
  }
  return <ComparisonView data={state.data} videoId={videoId} />;
}

const colors = {yolo: '#23cfff', dmcount: '#ffc653', fused: '#c084fc'};

const mean = (values: number[]) => (values.length ? values.reduce((sum, v) => sum + v, 0) / values.length : null);

function ComparisonView({data, videoId}: {data: import('../../services/api').ModelComparisonResponse; videoId: string}) {
  const zones = useMemo(() => [...new Set(data.items.map((item) => item.zone))].sort(), [data.items]);
  const [zone, setZone] = useState(zones[0] ?? '');
  const rows = useMemo(
    () => data.items.filter((item) => item.zone === zone).sort((a, b) => a.window_start - b.window_start),
    [data.items, zone],
  );
  const [selected, setSelected] = useState<number | null>(rows[0]?.window_start ?? null);
  // 구역을 바꾸면 선택 시점을 그 구역의 첫 윈도우로 되돌린다 — effect+setState로 하면
  // 렌더가 한 번 더 걸리니, React가 권장하는 "렌더 중 상태 보정" 패턴으로 처리한다.
  const [resetFor, setResetFor] = useState(zone);
  if (zone !== resetFor) {
    setResetFor(zone);
    setSelected(rows[0]?.window_start ?? null);
  }
  const selectedRow = rows.find((row) => row.window_start === selected) ?? null;
  // YOLO 박스 밖 잔여가 저장된 결과에서만 "YOLO + DM-Count" 선·열을 보여준다(예전 결과는 null).
  const hasFused = rows.some((row) => row.fused_count !== null);
  const summary = useMemo(() => {
    const paired = rows.filter((row) => row.yolo_count !== null && row.fused_count !== null);
    return {
      yolo: mean(paired.map((row) => row.yolo_count as number)),
      fused: mean(paired.map((row) => row.fused_count as number)),
      residual: mean(paired.map((row) => row.dmcount_residual_count as number)),
    };
  }, [rows]);

  const video = useRef<HTMLVideoElement>(null);
  const seekTo = (windowStart: number) => {
    setSelected(windowStart);
    const v = video.current;
    if (v) v.currentTime = windowStart;
  };
  useEffect(() => {
    const v = video.current;
    if (!v) return;
    // 영상이 재생되는 동안 현재 시점이 속한 10초 윈도우를 자동으로 따라간다(VideoZoneCounts와 같은 패턴).
    const update = () => {
      if (!rows.length) return;
      const windowStart = rows.reduce((closest, row) => (row.window_start <= v.currentTime ? row.window_start : closest), rows[0].window_start);
      setSelected(windowStart);
    };
    v.addEventListener('timeupdate', update);
    v.addEventListener('seeked', update);
    return () => { v.removeEventListener('timeupdate', update); v.removeEventListener('seeked', update); };
  }, [rows]);

  return (
    <section aria-label="YOLO·DM-Count 모델 비교" className="space-y-5">
      <div className="rounded-lg border border-amber-500/40 bg-surface p-4 text-xs text-ink-soft">
        <p className="font-semibold text-warn">DM-Count는 참고용 보조 추정치입니다 (정확도 미검증 — {data.accuracy_status})</p>
        <p className="mt-1 text-muted">{data.note}</p>
      </div>

      <div className="rounded-lg border border-line bg-surface p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 className="text-sm font-bold text-ink">시간대별 인원 추정 비교</h3>
          <label className="text-xs text-ink-soft">비교 구역 <select aria-label="비교 구역" value={zone} onChange={(event) => setZone(event.target.value)} className="ml-2 rounded-sm border border-line bg-surface px-2 py-1">{zones.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
        </div>
        {!rows.length ? <p className="py-8 text-center text-sm text-muted">이 구역에는 비교할 윈도우가 없습니다.</p> : <div aria-label="YOLO·DM-Count 비교 그래프" className="mt-4 h-60 w-full"><ResponsiveContainer width="100%" height="100%"><LineChart data={rows} margin={{top: 10, right: 20, left: 0, bottom: 5}}>
          <CartesianGrid stroke="var(--color-line)" vertical={false} />
          <XAxis dataKey="window_start" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(value) => `${value}초`} tick={{fill: 'var(--color-muted)', fontSize: 11}} />
          <YAxis allowDecimals unit="명" tick={{fill: 'var(--color-muted)', fontSize: 11}} />
          <Tooltip labelFormatter={(value) => `${value}초`} />
          <Legend />
          {selected !== null && <ReferenceLine x={selected} stroke="var(--color-ink-soft)" strokeDasharray="2 3" />}
          <Line dataKey="yolo_count" name="YOLO" type="linear" stroke={colors.yolo} strokeWidth={2} dot activeDot={{r: 5, onClick: (_, payload) => seekTo((payload as unknown as {payload: {window_start: number}}).payload.window_start)}} connectNulls={false} isAnimationActive={false} />
          {hasFused && <Line dataKey="fused_count" name="YOLO + DM-Count (보완)" type="linear" stroke={colors.fused} strokeWidth={2} dot={false} activeDot={{r: 5, onClick: (_, payload) => seekTo((payload as unknown as {payload: {window_start: number}}).payload.window_start)}} connectNulls={false} isAnimationActive={false} />}
          <Line dataKey="dmcount_estimated_count" name="DM-Count (참고)" type="linear" stroke={colors.dmcount} strokeWidth={2} strokeDasharray="5 4" dot activeDot={{r: 5, onClick: (_, payload) => seekTo((payload as unknown as {payload: {window_start: number}}).payload.window_start)}} isAnimationActive={false} />
        </LineChart></ResponsiveContainer></div>}
        {hasFused && summary.yolo !== null && summary.fused !== null && summary.residual !== null && <div className="mt-4 space-y-2 text-xs text-ink-soft">
          <dl className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div className="rounded-sm border border-line px-3 py-2"><dt className="text-muted">YOLO만 (평균)</dt><dd className="mt-1 text-base font-bold text-ink">{summary.yolo.toFixed(2)}명</dd></div>
            <div className="rounded-sm border border-line px-3 py-2"><dt className="text-muted">YOLO + DM-Count (평균)</dt><dd className="mt-1 text-base font-bold text-ink">{summary.fused.toFixed(2)}명</dd></div>
            <div className="rounded-sm border border-line px-3 py-2"><dt className="text-muted">DM-Count가 더한 인원 (평균)</dt><dd className="mt-1 text-base font-bold text-ink">+{summary.residual.toFixed(2)}명{summary.yolo > 0 ? ` (+${((summary.residual / summary.yolo) * 100).toFixed(0)}%)` : ''}</dd></div>
          </dl>
          <p className="text-muted">YOLO + DM-Count = YOLO 인원 + YOLO 사람 박스(10% 여유) 밖에 남은 DM-Count 밀도 합. YOLO가 놓친 먼 곳·겹친 사람을 채우려는 보완 추정이지만, 박스 밖에서 DM-Count가 사람으로 잘못 본 무늬(꽃·간판·조명·젖은 바닥의 빛 반사 등)도 함께 더해집니다.</p>
        </div>}
      </div>

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <div className="overflow-hidden rounded-lg border border-line bg-surface">
          <h3 className="px-5 py-4 text-sm font-bold text-ink">동일 시점 서버 영상</h3>
          <video ref={video} aria-label="모델 비교 동기화 영상" src={api.getUri({url: `/api/videos/${encodeURIComponent(videoId)}/stream`})} controls playsInline preload="metadata" className="block w-full" />
          <p className="px-5 py-3 text-xs text-muted">그래프의 점 또는 아래 표의 행을 클릭하면 그 시점으로 이동합니다. 영상을 재생하면 선택 시점이 따라갑니다.</p>
        </div>

        {/* key로 윈도우가 바뀔 때마다 새로 마운트해 이전 이미지의 onError 상태(failed)가 새 이미지에 남지 않게 한다. */}
        <HeatmapPanel key={selectedRow?.window_start ?? 'none'} row={selectedRow} />
      </div>

      <div className="overflow-hidden rounded-lg border border-line bg-surface">
        <h3 className="px-5 py-4 text-sm font-bold text-ink">윈도우별 비교표</h3>
        <div className="max-h-80 overflow-auto"><table className="w-full text-left text-xs text-ink-soft"><thead className="bg-sunken text-muted"><tr>{['시점', 'YOLO 평균 인원', ...(hasFused ? ['YOLO + DM-Count'] : []), 'DM-Count 추정', '차이', '추론 시간'].map((title) => <th key={title} scope="col" className="whitespace-nowrap px-4 py-3">{title}</th>)}</tr></thead><tbody>{rows.map((row) => {
          const diff = row.yolo_count === null ? null : row.dmcount_estimated_count - row.yolo_count;
          return <tr key={row.window_start} onClick={() => seekTo(row.window_start)} aria-current={row.window_start === selected} className={`cursor-pointer border-t border-line ${row.window_start === selected ? 'bg-sunken' : ''}`}>
            <td className="whitespace-nowrap px-4 py-3">{row.window_start}초</td>
            <td className="px-4 py-3">{row.yolo_count === null ? '동일 시점 YOLO 데이터 없음' : `${row.yolo_count.toLocaleString('ko-KR')}명`}</td>
            {hasFused && <td className="px-4 py-3">{row.fused_count === null ? '—' : `${row.fused_count.toLocaleString('ko-KR')}명 (+${(row.dmcount_residual_count ?? 0).toLocaleString('ko-KR')})`}</td>}
            <td className="px-4 py-3">{row.dmcount_estimated_count.toLocaleString('ko-KR')}명</td>
            <td className="px-4 py-3">{diff === null ? '—' : `${diff > 0 ? '+' : ''}${diff.toFixed(2)}명`}</td>
            <td className="whitespace-nowrap px-4 py-3">{(row.dmcount_inference_ms / 1000).toFixed(1)}초</td>
          </tr>;
        })}</tbody></table></div>
      </div>
    </section>
  );
}

function HeatmapPanel({row}: {row: import('../../services/api').ModelComparisonItem | null}) {
  const [failed, setFailed] = useState(false);
  return (
    <div className="overflow-hidden rounded-lg border border-line bg-surface">
      <h3 className="px-5 py-4 text-sm font-bold text-ink">선택 시점 DM-Count 밀도 히트맵</h3>
      {!row ? <p className="px-5 pb-5 text-sm text-muted">비교표에서 시점을 선택하세요.</p>
        : !row.density_map_path ? <p className="px-5 pb-5 text-sm text-muted">이 윈도우는 저장된 히트맵 이미지가 없습니다.</p>
        : failed ? <p role="alert" className="px-5 pb-5 text-sm text-danger">히트맵 이미지를 불러오지 못했습니다.</p>
        : <img src={api.getUri({url: row.density_map_path})} alt={`${row.window_start}초 ${row.zone} 구역 DM-Count 밀도 히트맵`} onError={() => setFailed(true)} className="block w-full" />}
      <p className="px-5 py-3 text-xs text-muted">색은 모델 출력 해상도 기준 상대적 밀도이며, 실제 인원수가 아닙니다.</p>
    </div>
  );
}
