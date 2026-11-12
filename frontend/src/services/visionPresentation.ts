import type {VisionResultSummary, VisionZoneMetric} from './api';

export const formatMetric = (value: number | null | undefined) => value == null ? '데이터 없음' : value.toLocaleString('ko-KR', {maximumFractionDigits: 4});
export const relativeSeconds = (milliseconds: number) => `${formatMetric(milliseconds / 1000)}초`;

export function directionLabel(summary: VisionResultSummary, row: VisionZoneMetric, kind: 'flow' | 'individual'): string {
  if (kind === 'flow' && summary.analysis_profile === 'fast') return '미분석 (FAST)';
  const angle = kind === 'flow' ? row.flow_angle_deg : row.mean_angle_deg;
  if (kind === 'flow' && row.has_flow && row.low_confidence === true) return '신뢰 낮음';
  if (!(kind === 'flow' ? row.has_flow : row.has_individual_aggregate) || angle === null) return '데이터 없음';
  if (summary.direction_presentation.quality === 'rejected') return '사용 제한 방향값';
  const allowed = summary.roi_status === 'approved' && summary.roi_snapshot.approved_for_analysis === true
    && summary.direction_presentation.quality === 'validated' && summary.direction_presentation.confirmed_for_display
    && (kind === 'flow' ? row.flow_direction_confirmed && row.low_confidence === false : row.individual_direction_confirmed);
  return allowed ? `${formatMetric(angle)}° (검증된 추정값)` : summary.analysis_profile === 'fast' ? `${formatMetric(angle)}° (미검증 방향값)` : '미검증 방향값';
}

export function crowdSeries(metrics: VisionZoneMetric[]) {
  const regions = [...new Set(metrics.map((row) => row.region_key))].sort();
  const points = new Map<number, Record<string, number | null>>();
  for (const row of metrics) {
    if (!points.has(row.video_offset_ms)) points.set(row.video_offset_ms, Object.fromEntries([
      ['video_offset_ms', row.video_offset_ms], ...regions.map((_, index) => [`zone${index}`, null]),
    ]));
    points.get(row.video_offset_ms)![`zone${regions.indexOf(row.region_key)}`] = row.has_observation ? row.mean_person_count : null;
  }
  return {regions, points: [...points.values()].sort((a, b) => a.video_offset_ms! - b.video_offset_ms!)};
}
