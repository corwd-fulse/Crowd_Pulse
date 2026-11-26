// 행사장 지도 '구역 단위 예측 히트맵': 지도 구역에 연결된 영상 ROI의 시계열 예측(10·30·60초 뒤 인원)을 색으로 칠한다.
// 예측 모델은 구역별 인원만 예측하므로 구역 안의 픽셀 분포를 그리지 않는다(근거 없는 값 금지).
import type {VideoForecastModel, VideoForecastResponse} from './api';

export const FORECAST_HORIZONS = [10, 30, 60] as const;
const CANDIDATES: VideoForecastModel[] = ['lightgbm_pre', 'lstm_pre', 'lightgbm', 'ridge_ar'];
// 낮음 -> 높음 순차 색(밝은 노랑 -> 주황 -> 빨강). 병목 정책 색과 섞이지 않게 별도 범례로 쓴다.
const STOPS: [number, [number, number, number]][] = [[0, [254, 240, 138]], [0.5, [249, 115, 22]], [1, [220, 38, 38]]];

export interface ZoneForecast {
  predicted: number | null;      // H초 뒤 예측 인원(10초 평균 기준)
  model: VideoForecastModel | null;
  modelLabel: string;
  reference: number | null;      // 색 기준 인원: 수용 인원(설정 시) 또는 이 영상 그 구역 최대 관측 인원
  referenceKind: 'capacity' | 'video_max' | null;
  ratio: number | null;          // predicted / reference (0~1로 자름)
  color: string;
  reason: string;
}

const LABEL: Record<VideoForecastModel, string> = {
  persistence: '직전값', ridge_ar: 'Ridge AR', lightgbm: 'LightGBM(영상 내 학습)', lightgbm_pre: 'LightGBM(미리 학습)', lstm_pre: 'LSTM(미리 학습)',
};

/** 이 예측 시간에 직전값보다 오차가 작은 모델 중 MAE 최저. 없으면 직전값(모델을 쓰지 않음). */
export function pickModel(data: VideoForecastResponse, horizon: number): VideoForecastModel {
  const rows = data.backtest.filter((r) => r.horizon_s === horizon && CANDIDATES.includes(r.model as VideoForecastModel) && r.skill > 0);
  rows.sort((a, b) => a.mae - b.mae);
  return (rows[0]?.model as VideoForecastModel | undefined) ?? 'persistence';
}

export function heatColor(ratio: number): string {
  const r = Math.min(1, Math.max(0, ratio));
  for (let i = 1; i < STOPS.length; i++) {
    const [p1, c1] = STOPS[i], [p0, c0] = STOPS[i - 1];
    if (r <= p1) {
      const t = (r - p0) / (p1 - p0);
      return `rgb(${c0.map((v, k) => Math.round(v + (c1[k] - v) * t)).join(',')})`;
    }
  }
  return `rgb(${STOPS[STOPS.length - 1][1].join(',')})`;
}

const EMPTY = (reason: string): ZoneForecast => ({predicted: null, model: null, modelLabel: '—', reference: null, referenceKind: null, ratio: null, color: '#738398', reason});

/** 영상 시각 time(초)에서 본 H초 뒤 예측. 예측 점은 target_sec(= 마지막 관측 초 + H) 기준으로 저장돼 있다. */
export function zoneForecast(data: VideoForecastResponse | null, roi: string | undefined, time: number, horizon: number, capacity: number | null): ZoneForecast {
  if (!roi) return EMPTY('영상 ROI 연결 필요');
  if (!data) return EMPTY('이 작업의 시계열 예측 결과 없음');
  const rows = data.series.filter((p) => p.zone === roi);
  if (!rows.length) return EMPTY(`예측에 ${roi} 구역 없음`);
  const model = pickModel(data, horizon);
  const target = Math.floor(time) + horizon;
  const point = rows.find((p) => p.sec === target);
  const predicted = point ? (point[model] ?? null) : null;
  const videoMax = Math.max(0, ...rows.map((p) => p.actual ?? 0));
  const reference = capacity && capacity > 0 ? capacity : videoMax > 0 ? videoMax : null;
  const referenceKind = capacity && capacity > 0 ? 'capacity' : reference ? 'video_max' : null;
  if (predicted === null) return {...EMPTY(point ? '이 시점은 학습 이력 부족으로 예측 없음' : '이 시점 예측 없음(영상 시작 직후·끝 이후)'), model, modelLabel: LABEL[model], reference, referenceKind};
  const value = Math.max(0, predicted);
  const ratio = reference ? Math.min(1, value / reference) : null;
  return {predicted: value, model, modelLabel: LABEL[model], reference, referenceKind, ratio, color: ratio === null ? '#738398' : heatColor(ratio),
    reason: model === 'persistence' ? '이 예측 시간에는 직전값보다 나은 모델이 없어 직전값을 표시' : `${LABEL[model]} · 직전값보다 오차가 작은 모델 중 최저`};
}
