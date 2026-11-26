import {describe, expect, it} from 'vitest';
import {heatColor, pickModel, zoneForecast} from '../src/services/venueForecast';
import type {VideoForecastResponse} from '../src/services/api';

const data = (skill: Record<string, number>): VideoForecastResponse => ({
  analysis_job_id: 'j', source: 'yolo', sources: ['yolo'], horizon_s: 30, horizons: [10, 30, 60], zones: ['A'], seconds: 100, target: 'mean10',
  models: ['persistence', 'ridge_ar', 'lightgbm', 'lightgbm_pre', 'lstm_pre'],
  backtest: Object.entries(skill).map(([model, s], i) => ({horizon_s: 30, model, mae: 2 - i * 0.1, rmse: 1, skill: s, n: 10})),
  series: [
    {zone: 'A', sec: 40, split: 'backtest', actual: 20, persistence: 8, lightgbm_pre: 12, lstm_pre: 13, ridge_ar: -2},
    {zone: 'A', sec: 45, split: 'backtest', actual: 5, persistence: 5, lightgbm_pre: null, lstm_pre: 6},
  ],
  level_segments: [], level_seconds: {}, level_events: [], iforest: null, minutes: null, note: '',
} as unknown as VideoForecastResponse);

describe('venue forecast heatmap', () => {
  it('uses the lowest-MAE model that beats persistence, else persistence', () => {
    expect(pickModel(data({persistence: 0, lightgbm_pre: 0.1, lstm_pre: 0.2}), 30)).toBe('lstm_pre');
    expect(pickModel(data({persistence: 0, lightgbm_pre: -0.1, lstm_pre: -0.3}), 30)).toBe('persistence');
  });

  it('reads the point whose target second is floor(time) + horizon', () => {
    const f = zoneForecast(data({persistence: 0, lightgbm_pre: 0.2}), 'A', 10.7, 30, null);
    expect(f.predicted).toBe(12);
    expect(f.referenceKind).toBe('video_max');
    expect(f.ratio).toBeCloseTo(12 / 20);
  });

  it('prefers manager capacity as the color reference and clamps', () => {
    const f = zoneForecast(data({persistence: 0, lightgbm_pre: 0.2}), 'A', 10, 30, 10);
    expect(f.referenceKind).toBe('capacity');
    expect(f.ratio).toBe(1);
    expect(f.color).toBe(heatColor(1));
  });

  it('never invents values', () => {
    expect(zoneForecast(null, 'A', 10, 30, null).predicted).toBeNull();
    expect(zoneForecast(data({persistence: 0, lightgbm_pre: 0.2}), undefined, 10, 30, null).reason).toMatch(/ROI/);
    expect(zoneForecast(data({persistence: 0, lightgbm_pre: 0.2}), 'A', 15, 30, null).predicted).toBeNull(); // 학습 이력 부족
    expect(zoneForecast(data({persistence: 0, lightgbm_pre: 0.2}), 'A', 99, 30, null).predicted).toBeNull();
  });
});
