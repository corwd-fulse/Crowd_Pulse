import {useState} from 'react';
import {CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis} from 'recharts';
import type {AnalysisJobResponse, VisionResultBundle} from '../../services/api';
import {api} from '../../services/api';
import {crowdSeries, directionLabel, formatMetric, relativeSeconds} from '../../services/visionPresentation';
import {useVisionResult} from '../../hooks/useVisionResult';
import StartCard from './StartCard';

const colors = ['#23cfff', '#ffc653', '#23e3bb', '#ff8db3', '#ab9bff'];

export default function AnalysisResults({job}: {job: AnalysisJobResponse | null}) {
  const {state, retry} = useVisionResult(job);
  if (!job) return <p className="rounded-lg border border-line bg-surface p-5 text-sm text-muted">조회할 분석 작업이 없습니다. 영상을 업로드하고 분석을 요청하세요.</p>;
  if (job.status === 'failed') return <div role="alert" className="rounded-lg border border-rose-500/40 bg-surface p-5 text-sm text-ink"><p className="font-semibold">분석 실패</p><p className="mt-2 break-words">{job.error_message || '상세 오류 메시지가 없습니다.'}</p><p className="mt-2 text-muted">다시 분석하려면 새 작업을 요청하세요.</p></div>;
  if (job.status === 'cancelled') return <p role="status" className="rounded-lg border border-line bg-surface p-5 text-sm text-muted">취소된 분석 작업입니다.</p>;
  if (job.status === 'queued' || job.status === 'running') return <p role="status" className="rounded-lg border border-line bg-surface p-5 text-sm text-muted">{job.status === 'queued' ? '분석 대기 중입니다. 별도 Worker가 작업을 처리하면 결과가 표시됩니다.' : '분석 진행 중입니다. 작업 상태를 5초마다 확인합니다.'}</p>;
  if (state.status === 'loading' || state.status === 'idle') return <p role="status" className="rounded-lg border border-line bg-surface p-5 text-sm text-muted">실제 분석 결과를 불러오는 중…</p>;
  if (state.status === 'error') return <div role="alert" className="rounded-lg border border-line bg-surface p-5 text-sm text-ink"><p>{state.message}</p><button type="button" onClick={retry} className="mt-3 rounded-sm border border-line px-3 py-2">결과 다시 조회</button></div>;
  return <ResultView key={state.data.summary.analysis_job_id} data={state.data} />;
}

function ResultView({data}: {data: VisionResultBundle}) {
  const {summary} = data;
  const fast = summary.analysis_profile === 'fast';
  const [region, setRegion] = useState('');
  const [page, setPage] = useState(0);
  const metrics = data.metrics.filter((row) => !region || row.region_key === region);
  const scores = data.scores.filter((row) => !region || row.region_key === region);
  const regions = [...new Set([...summary.roi_snapshot.zones.map((zone) => zone.name), ...data.metrics.map((row) => row.region_key), ...data.scores.map((row) => row.region_key)])];
  const series = crowdSeries(metrics);
  const values = metrics.filter((row) => row.has_observation).map((row) => row.mean_person_count).filter((value): value is number => value !== null);
  const total = (key: 'inflow' | 'outflow') => {
    const entries = metrics.filter((row) => row.has_observation).map((row) => row[key]).filter((value): value is number => value !== null);
    return entries.length ? entries.reduce((sum, value) => sum + value, 0) : null;
  };
  const peakScore = scores.length ? scores.reduce((peak, row) => Math.max(peak, row.score), 0) : null;
  const tableRows = metrics.slice(page * 20, (page + 1) * 20);
  const unapproved = summary.roi_status !== 'approved' || !summary.roi_snapshot.approved_for_analysis;
  return (
    <section aria-label="실제 AI 분석 결과" data-source="analysis-result" className="space-y-5">
      <div className="rounded-lg border border-line bg-surface p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-sm font-bold text-ink">저장된 AI 분석 결과</h2>
          <label className="text-xs text-ink-soft">조회 구역 <select aria-label="결과 구역" value={region} onChange={(event) => {setRegion(event.target.value); setPage(0);}} className="ml-2 rounded-sm border border-line bg-surface px-2 py-1"><option value="">전체 구역</option>{regions.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
        </div>
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          <span className="rounded-sm border border-brand px-2 py-1 text-brand">{fast?'FAST 빠른 분석 완료':'FULL 정밀 분석 완료'}</span>
          <span className="rounded-sm border border-brand px-2 py-1 text-brand">{summary.analysis_mode === 'test' ? '테스트 분석 결과' : '운영 분석 결과'}</span>
          <span className="rounded-sm border border-line px-2 py-1 text-ink-soft">{unapproved ? '미승인 ROI' : '승인 ROI'}</span>
          <span className="rounded-sm border border-line px-2 py-1 text-ink-soft">{summary.direction_presentation.quality === 'unvalidated' ? '방향 품질 미검증' : summary.direction_presentation.quality === 'rejected' ? '방향 사용 제한' : '방향 품질 검증 기록 있음'}</span>
        </div>
        <p className="mt-3 break-all text-xs text-muted">작업 {summary.analysis_job_id} · 버전 {summary.direction_presentation.algorithm_version}</p>
        <p className="mt-1 break-all text-xs text-muted">분석 대상 영상 ID {summary.video_id}</p>
        <p className="mt-1 text-xs text-muted">분석 완료 {new Date(summary.analysis_finished_at).toLocaleString('ko-KR')} · 아래 시간은 영상 시작 기준 상대 시간입니다.</p>
        {!summary.direction_presentation.confirmed_for_display && <p className="mt-2 text-xs text-warn">방향값은 검증 전 또는 사용 제한 데이터입니다. 결과 생성 성공은 방향 품질 승인을 의미하지 않습니다.</p>}
      </div>

      <div className="summary-grid grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StartCard featured title="구역 최고 평균 인원" value={values.length ? `${formatMetric(values.reduce((peak, value) => Math.max(peak, value), 0))}명` : '—'} description="구역·초별 평균 중 최댓값" />
        <StartCard title="유입 집계 합계" value={formatMetric(total('inflow'))} description="선택 구역·영상 기간 합계 / 고유 인원 아님" />
        <StartCard title="유출 집계 합계" value={formatMetric(total('outflow'))} description="선택 구역·영상 기간 합계 / 고유 인원 아님" />
        <StartCard color="amber" title="최고 병목 점수" value={fast?'미분석':peakScore === null ? '—' : `${formatMetric(peakScore)} / 100`} description={fast?'FAST는 병목 위험도를 평가하지 않습니다':'병목 연구 지표 / 혼잡도 퍼센트 아님'} />
      </div>

      <div className="rounded-lg border border-line bg-surface p-5">
        <h3 className="text-sm font-bold text-ink">구역별 시간대별 평균 인원</h3>
        <p className="mt-1 text-xs text-muted">영상 상대 시간 · 소수 평균 인원 · 예측 데이터 없음</p>
        {!values.length ? <p className="py-8 text-center text-sm text-muted">표시할 인원 관측 데이터가 없습니다.</p> : <div aria-label="평균 인원 그래프" className="mt-4 h-60 w-full"><ResponsiveContainer width="100%" height="100%"><LineChart data={series.points} margin={{top: 10, right: 20, left: 0, bottom: 5}}><CartesianGrid stroke="var(--color-line)" vertical={false}/><XAxis dataKey="video_offset_ms" type="number" domain={['dataMin', 'dataMax']} tickFormatter={relativeSeconds} tick={{fill: 'var(--color-muted)', fontSize: 11}}/><YAxis allowDecimals unit="명" tick={{fill: 'var(--color-muted)', fontSize: 11}}/><Tooltip labelFormatter={(value) => relativeSeconds(Number(value))}/><Legend/>{series.regions.map((name, index) => <Line key={name} dataKey={`zone${index}`} name={name} type="linear" stroke={colors[index % colors.length]} strokeWidth={2} dot={false} connectNulls={false} isAnimationActive={false}/>)}</LineChart></ResponsiveContainer></div>}
      </div>

      <div className="overflow-hidden rounded-lg border border-line bg-surface">
        <h3 className="px-5 py-4 text-sm font-bold text-ink">유입·유출 및 방향 품질</h3>
        {!metrics.length ? <p className="px-5 pb-5 text-sm text-muted">저장된 구역 지표가 없습니다.</p> : <>
          <div className="overflow-x-auto"><table className="w-full text-left text-xs text-ink-soft"><thead className="bg-sunken text-muted"><tr>{['영상 시간', '구역', '평균 인원', '유입', '유출', 'Optical Flow 방향', '개인 방향 평균', '방향 표본 수'].map((title) => <th key={title} scope="col" className="whitespace-nowrap px-4 py-3">{title}</th>)}</tr></thead><tbody>{tableRows.map((row) => <tr key={`${row.region_key}-${row.video_offset_ms}`} className="border-t border-line"><td className="whitespace-nowrap px-4 py-3">{relativeSeconds(row.video_offset_ms)}</td><td className="px-4 py-3">{row.region_key}</td><td className="px-4 py-3">{formatMetric(row.has_observation ? row.mean_person_count : null)}</td><td className="px-4 py-3">{formatMetric(row.has_observation ? row.inflow : null)}</td><td className="px-4 py-3">{formatMetric(row.has_observation ? row.outflow : null)}</td><td className="whitespace-nowrap px-4 py-3">{directionLabel(summary, row, 'flow')}</td><td className="whitespace-nowrap px-4 py-3">{directionLabel(summary, row, 'individual')}</td><td className="px-4 py-3">{formatMetric(row.direction_sample_count)}</td></tr>)}</tbody></table></div>
          <div className="flex items-center justify-between border-t border-line px-5 py-3 text-xs text-muted"><span>{page + 1} / {Math.ceil(metrics.length / 20)}쪽 · 전체 {metrics.length}행</span><div className="flex gap-3"><button type="button" disabled={page === 0} onClick={() => setPage((value) => value - 1)} className="disabled:opacity-40">이전 지표</button><button type="button" disabled={(page + 1) * 20 >= metrics.length} onClick={() => setPage((value) => value + 1)} className="disabled:opacity-40">다음 지표</button></div></div>
        </>}
      </div>

      <div className="overflow-hidden rounded-lg border border-line bg-surface">
        <h3 className="px-5 py-4 text-sm font-bold text-ink">병목 점수·위험 등급</h3>
        {fast&&<p role="status" className="px-5 pb-4 text-sm text-warn">Optical Flow·병목 미분석 — FAST에서는 실행하지 않았습니다. 위험 없음 판정이 아닙니다. 영상 관리에서 FULL 정밀 분석을 별도로 요청하세요.</p>}
        <p className="px-5 pb-3 text-xs text-muted">점수는 연구용 지표입니다. 실제 사고·위험 확정 판정이 아닙니다.</p>
        {fast?null:!scores.length ? <p className="px-5 pb-5 text-sm text-muted">저장된 병목 점수가 없습니다. 빈 결과도 정상 분석 완료일 수 있습니다.</p> : <div className="max-h-80 overflow-auto"><table className="w-full text-left text-xs text-ink-soft"><thead className="bg-sunken"><tr>{['윈도우 시작', '구역', '점수 (0–100)', '위험 등급', '증가·둔화·불균형'].map((title) => <th key={title} scope="col" className="whitespace-nowrap px-4 py-3">{title}</th>)}</tr></thead><tbody>{scores.map((row) => <tr key={`${row.region_key}-${row.window_start_ms}`} className="border-t border-line"><td className="px-4 py-3">{relativeSeconds(row.window_start_ms)}</td><td className="px-4 py-3">{row.region_key}</td><td className="px-4 py-3">{formatMetric(row.score)}</td><td className={`px-4 py-3 ${row.severity === 'high' ? 'text-danger' : row.severity === 'medium' ? 'text-warn' : 'text-ok'}`}>{row.risk_level}</td><td className="px-4 py-3">{formatMetric(row.growth)} · {formatMetric(row.slowdown)} · {formatMetric(row.imbalance)}</td></tr>)}</tbody></table></div>}
      </div>

      <div className="rounded-lg border border-line bg-surface p-5 text-xs text-muted"><h3 className="mb-2 font-semibold text-ink">결과 CSV ({Object.keys(summary.artifacts).length}개)</h3><p className="mb-3">개인 방향 원시값과 reliable은 CSV에 보존됩니다. n_tracks는 고유 인원수가 아닌 표본 수이고 speed_px는 실제 속도가 아닙니다.</p><div className="flex flex-wrap gap-3">{Object.entries(summary.artifacts).map(([kind, artifact]) => <a key={kind} className="text-brand underline" href={api.getUri({url: artifact.download_url})}>{kind} · {artifact.rows.toLocaleString()}행</a>)}</div></div>
    </section>
  );
}
