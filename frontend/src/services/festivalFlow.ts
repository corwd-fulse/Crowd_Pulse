// 축제 전체 흐름: 카메라를 구역 사이의 '문'으로 보고 구역 인원 = 시작 인원 + 누적 유입 - 누적 유출 로 계산한다.
// 예측(10·30·60초 뒤)은 최근 RATE_WINDOW_S초 순유입 속도가 그대로 이어진다는 가정(일정 속도)이다.
import {api} from './api';

export const RATE_WINDOW_S = 60;

export interface GateCounts {job_id: string; seconds: number; plus: number[]; minus: number[]; axis: 'x' | 'y'; line: number; plus_means: string; minus_means: string}
export interface FlowCamera {id: string; name: string; job_id: string | null; axis: 'x' | 'y'; start_offset_s: number;
  plus: Record<string, number>; minus: Record<string, number>; meaning: string; gate: GateCounts | null}
export interface FestivalFlowConfig {venue_id: string; name: string; sync_note: string; zones: Record<string, {initial: number}>; cameras: FlowCamera[]}

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
  cameraActive: Record<string, [number, number] | null>; // 카메라별 [시작, 끝) 통합 초
}

/** 카메라 통과 수 -> 구역별 누적 순변화. 카메라 로컬 초 s 는 통합 초 start_offset_s + s 에 놓인다. */
export function buildTimeline(cfg: FestivalFlowConfig): FlowTimeline {
  const cams = cfg.cameras.filter((c) => c.gate);
  const length = Math.max(1, ...cams.map((c) => c.start_offset_s + c.gate!.seconds));
  const delta: Record<string, Float64Array> = {};
  for (const z of Object.keys(cfg.zones)) {
    const step = new Float64Array(length);
    for (const c of cams) {
      const g = c.gate!, wp = c.plus[z] ?? 0, wm = c.minus[z] ?? 0;
      if (!wp && !wm) continue;
      for (let s = 0; s < g.seconds; s++) {
        const t = c.start_offset_s + s;
        if (t >= 0 && t < length) step[t] += g.plus[s] * wp + g.minus[s] * wm;
      }
    }
    for (let t = 1; t < length; t++) step[t] += step[t - 1];
    delta[z] = step;
  }
  const cameraActive: FlowTimeline['cameraActive'] = {};
  for (const c of cfg.cameras) cameraActive[c.id] = c.gate ? [c.start_offset_s, c.start_offset_s + c.gate.seconds] : null;
  return {length, delta, cameraActive};
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
