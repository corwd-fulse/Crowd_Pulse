// 축제 전체 흐름: 카메라를 구역 사이의 '문'으로 보고 구역 인원 = 시작 인원 + 누적 유입 - 누적 유출 로 계산한다.
// 예측(10·30·60초 뒤)은 최근 RATE_WINDOW_S초 순유입 속도가 그대로 이어진다는 가정(일정 속도)이다.
import {api} from './api';

export const RATE_WINDOW_S = 60;

// 혼잡도 3단계: 기준 인원(관리자 입력 수용 인원, 없으면 이 타임라인 최대 인원) 대비 비율
export const CONGESTION_LEVELS = [
  {max: 0.5, label: '여유', color: '#22c55e'},
  {max: 0.8, label: '주의', color: '#f59e0b'},
  {max: Infinity, label: '위험', color: '#dc2626'},
] as const;
export function congestionLevel(count: number, reference: number) {
  const ratio = reference > 0 ? count / reference : 0;
  return {ratio, ...CONGESTION_LEVELS.find((l) => ratio < l.max)!};
}

export interface GateCounts {job_id: string; seconds: number; plus: number[]; minus: number[]; axis: 'x' | 'y'; line: number; plus_means: string; minus_means: string;
  exit_sides?: Record<ExitSide, number[]>}
export type ExitSide = 'left' | 'center' | 'right';
export const EXIT_SIDES: ExitSide[] = ['left', 'center', 'right'];
export interface FlowCamera {id: string; name: string; job_id: string | null; axis: 'x' | 'y'; start_offset_s: number;
  plus: Record<string, number>; minus: Record<string, number>; meaning: string; gate: GateCounts | null}
// initial = 기본 시작 인원. auto_min_margin 이 있으면 실제 시작 인원 = max(initial, 측정 흐름이 0 아래로 안 내려가는 최소값 + margin)
export interface PhaseCamera {id: string; name: string; job_id: string; compare_to: string; axis: 'x' | 'y'; exit_zone?: string; gate: GateCounts | null}
export interface FlowPhase {label: string; purpose: string; cameras: PhaseCamera[]}
export interface FestivalFlowConfig {venue_id: string; name: string; sync_note: string; initial_note?: string;
  zones: Record<string, {initial: number; auto_min_margin?: number; basis?: string}>; cameras: FlowCamera[];
  phases?: Record<'pre' | 'post', FlowPhase>; exit_labels?: Record<ExitSide, string>; exit_labels_note?: string}

/** 구역별 실제 시작 인원과 그 근거(카메라가 추가되면 자동으로 다시 계산된다). */
export function startCounts(cfg: FestivalFlowConfig, tl: FlowTimeline): Record<string, {count: number; minDelta: number; minAt: number; auto: boolean}> {
  const out: Record<string, {count: number; minDelta: number; minAt: number; auto: boolean}> = {};
  for (const [z, zc] of Object.entries(cfg.zones)) {
    const d = tl.delta[z];
    let minDelta = 0, minAt = 0;
    if (d) for (let t = 0; t < d.length; t++) if (d[t] < minDelta) { minDelta = d[t]; minAt = t; }
    const needed = Math.ceil(-minDelta) + (zc.auto_min_margin ?? 0);
    const auto = zc.auto_min_margin !== undefined && needed > zc.initial;
    out[z] = {count: auto ? needed : zc.initial, minDelta, minAt, auto};
  }
  return out;
}

export async function getFestivalFlow(venueId: string, signal?: AbortSignal): Promise<FestivalFlowConfig | null> {
  try {
    return (await api.get<FestivalFlowConfig>(`/api/venues/${encodeURIComponent(venueId)}/festival-flow`, {signal})).data;
  } catch {
    return null;
  }
}

export interface FlowTimeline {
  length: number;                         // 통합 타임라인 길이(초)
  delta: Record<string, Float64Array>;   // 구역별 누적 순변화(시작 인원 제외), index = 통합 초
  inflow: Record<string, Float64Array>;  // 구역별 누적 들어옴
  outflow: Record<string, Float64Array>; // 구역별 누적 나감
  cameraActive: Record<string, [number, number] | null>; // 카메라별 [시작, 끝) 통합 초
}

/** 카메라 통과 수 -> 구역별 누적 순변화. 카메라 로컬 초 s 는 통합 초 start_offset_s + s 에 놓인다. */
export function buildTimeline(cfg: FestivalFlowConfig): FlowTimeline {
  const cams = cfg.cameras.filter((c) => c.gate);
  const length = Math.max(1, ...cams.map((c) => c.start_offset_s + c.gate!.seconds));
  const delta: Record<string, Float64Array> = {}, inflow: Record<string, Float64Array> = {}, outflow: Record<string, Float64Array> = {};
  for (const z of Object.keys(cfg.zones)) {
    const step = new Float64Array(length), inn = new Float64Array(length), out = new Float64Array(length);
    for (const c of cams) {
      const g = c.gate!, wp = c.plus[z] ?? 0, wm = c.minus[z] ?? 0;
      if (!wp && !wm) continue;
      for (let s = 0; s < g.seconds; s++) {
        const t = c.start_offset_s + s;
        if (t < 0 || t >= length) continue;
        for (const v of [g.plus[s] * wp, g.minus[s] * wm]) { step[t] += v; if (v > 0) inn[t] += v; else out[t] -= v; }
      }
    }
    for (let t = 1; t < length; t++) { step[t] += step[t - 1]; inn[t] += inn[t - 1]; out[t] += out[t - 1]; }
    delta[z] = step; inflow[z] = inn; outflow[z] = out;
  }
  const cameraActive: FlowTimeline['cameraActive'] = {};
  for (const c of cfg.cameras) cameraActive[c.id] = c.gate ? [c.start_offset_s, c.start_offset_s + c.gate.seconds] : null;
  return {length, delta, inflow, outflow, cameraActive};
}

export interface ZoneFlow {raw: number; shown: number; belowZero: boolean; rate: number; predicted: number | null; change: number | null}

/** time 초의 구역 인원과 H초 뒤 예측. raw<0 이면 0으로 보여 주고 belowZero 표시(시작 인원이 실제보다 적을 수 있음). */
export function zoneFlowAt(tl: FlowTimeline, zone: string, initial: number, time: number, horizon: number): ZoneFlow {
  const d = tl.delta[zone];
  const t = Math.max(0, Math.min(tl.length - 1, Math.floor(time)));
  const now = d ? d[t] : 0;
  const back = Math.max(0, t - RATE_WINDOW_S);
  const span = t - back;
  const rate = d && span > 0 ? (d[t] - d[back]) / span : 0;          // 명/초
  const raw = initial + now;
  const predictedRaw = horizon ? raw + rate * horizon : null;
  return {raw, shown: Math.max(0, raw), belowZero: raw < 0, rate,
    predicted: predictedRaw === null ? null : Math.max(0, predictedRaw), change: predictedRaw === null ? null : predictedRaw - raw};
}

/** 카메라 하나의 최근 창 통과 속도(명/분). */
export function cameraRates(cam: FlowCamera, time: number): {plus: number; minus: number} | null {
  const g = cam.gate;
  if (!g) return null;
  const s = Math.floor(time) - cam.start_offset_s;
  if (s < 0 || s >= g.seconds) return null;
  const a = Math.max(0, s - RATE_WINDOW_S + 1), n = s - a + 1;
  let p = 0, m = 0;
  for (let i = a; i <= s; i++) { p += g.plus[i]; m += g.minus[i]; }
  return {plus: p / n * 60, minus: m / n * 60};
}


/** 영상 전체 평균 통과 속도(명/분). plus/minus 의미는 카메라 축에 따른다(x: 오른쪽/왼쪽, y: 카메라 쪽=나감 / 멀어짐=들어감). */
export function averageRates(g: GateCounts | null): {plus: number; minus: number} | null {
  if (!g || !g.seconds) return null;
  const sum = (a: number[]) => a.reduce((x, y) => x + y, 0);
  return {plus: sum(g.plus) / g.seconds * 60, minus: sum(g.minus) / g.seconds * 60};
}

/** 나간 사람이 화면 어느 쪽으로 빠졌는지 수와 비율. */
export function exitShares(g: GateCounts | null): {counts: Record<ExitSide, number>; shares: Record<ExitSide, number>; total: number} | null {
  if (!g?.exit_sides) return null;
  const counts = Object.fromEntries(EXIT_SIDES.map((k) => [k, g.exit_sides![k].reduce((x, y) => x + y, 0)])) as Record<ExitSide, number>;
  const total = EXIT_SIDES.reduce((n, k) => n + counts[k], 0);
  const shares = Object.fromEntries(EXIT_SIDES.map((k) => [k, total ? counts[k] / total : 0])) as Record<ExitSide, number>;
  return {counts, shares, total};
}

/** 남은 인원이 지금 순유출 속도(명/분)로 다 빠지는 데 걸리는 분. 순유출이 0 이하면 null(줄지 않음). */
export function minutesToEmpty(remaining: number, netOutPerMin: number): number | null {
  return netOutPerMin > 0 ? remaining / netOutPerMin : null;
}

/** 카메라 영상 시각 time(초)까지의 누적 통과 수. 출입구 카메라(axis y)에서 plus = 나감, minus = 들어감. */
export function cumulativeAt(g: GateCounts | null, time: number): {exits: number; entries: number; sides: Record<ExitSide, number> | null} | null {
  if (!g) return null;
  const t = Math.max(0, Math.min(g.seconds - 1, Math.floor(time)));
  let exits = 0, entries = 0;
  for (let i = 0; i <= t; i++) { exits += g.plus[i]; entries += g.minus[i]; }
  const sides = g.exit_sides ? Object.fromEntries(EXIT_SIDES.map((k) => [k, g.exit_sides![k].slice(0, t + 1).reduce((a, b) => a + b, 0)])) as Record<ExitSide, number> : null;
  return {exits, entries, sides};
}

/** 두 축제 사이 통로 카메라(한 방향이 한 축제 -1, 다른 축제 +1). 방향별 누적과 H초 뒤 예상 이동 인원. */
export interface CorridorFlow {from: string; to: string; cumulative: number; perMin: number; predicted: Record<number, number>}
export function corridorFlows(cam: FlowCamera, time: number, horizons: readonly number[] = [10, 30, 60]): CorridorFlow[] {
  const g = cam.gate;
  if (!g) return [];
  const local = time - cam.start_offset_s;
  const cum = cumulativeAt(g, local), rate = cameraRates(cam, time);
  const out: CorridorFlow[] = [];
  for (const [dir, weights] of [['plus', cam.plus], ['minus', cam.minus]] as const) {
    const from = Object.keys(weights).find((z) => weights[z] < 0), to = Object.keys(weights).find((z) => weights[z] > 0);
    if (!from || !to) continue;
    const perMin = rate ? rate[dir] : 0;
    out.push({from, to, cumulative: cum ? (dir === 'plus' ? cum.exits : cum.entries) : 0, perMin,
      predicted: Object.fromEntries(horizons.map((h) => [h, perMin / 60 * h]))});
  }
  return out;
}
export const isCorridorCamera = (c: FlowCamera) => Object.keys(c.plus).length > 1 || Object.keys(c.minus).length > 1;
