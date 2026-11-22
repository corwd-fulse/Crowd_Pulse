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

const colors = {yolo: '#23cfff', dmcount: '#ffc653'};

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
          <Line dataKey="dmcount_estimated_count" name="DM-Count (참고)" type="linear" stroke={colors.dmcount} strokeWidth={2} strokeDasharray="5 4" dot activeDot={{r: 5, onClick: (_, payload) => seekTo((payload as unknown as {payload: {window_start: number}}).payload.window_start)}} isAnimationActive={false} />
        </LineChart></ResponsiveContainer></div>}
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
        <div className="max-h-80 overflow-auto"><table className="w-full text-left text-xs text-ink-soft"><thead className="bg-sunken text-muted"><tr>{['시점', 'YOLO 평균 인원', 'DM-Count 추정', '차이', '추론 시간'].map((title) => <th key={title} scope="col" className="whitespace-nowrap px-4 py-3">{title}</th>)}</tr></thead><tbody>{rows.map((row) => {
          const diff = row.yolo_count === null ? null : row.dmcount_estimated_count - row.yolo_count;
          return <tr key={row.window_start} onClick={() => seekTo(row.window_start)} aria-current={row.window_start === selected} className={`cursor-pointer border-t border-line ${row.window_start === selected ? 'bg-sunken' : ''}`}>
            <td className="whitespace-nowrap px-4 py-3">{row.window_start}초</td>
            <td className="px-4 py-3">{row.yolo_count === null ? '동일 시점 YOLO 데이터 없음' : `${row.yolo_count.toLocaleString('ko-KR')}명`}</td>
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
