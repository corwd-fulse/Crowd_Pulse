import {describe, expect, it} from 'vitest';
import {buildTimeline, cameraRates, zoneFlowAt} from '../src/services/festivalFlow';
import type {FestivalFlowConfig} from '../src/services/festivalFlow';

const cfg = (): FestivalFlowConfig => ({
  venue_id: 'main', name: 'f', sync_note: '', zones: {top: {initial: 35}, bottom: {initial: 35}},
  cameras: [
    {id: 'mid', name: 'mid', job_id: 'm', axis: 'x', start_offset_s: 0, plus: {top: 1, bottom: -1}, minus: {bottom: 1, top: -1}, meaning: '',
     gate: {job_id: 'm', seconds: 4, plus: [0, 2, 0, 0], minus: [1, 0, 0, 0], axis: 'x', line: 0.5, plus_means: 'right', minus_means: 'left'}},
    {id: 'bot', name: 'bot', job_id: 'b', axis: 'y', start_offset_s: 2, plus: {bottom: -1}, minus: {bottom: 1}, meaning: '',
     gate: {job_id: 'b', seconds: 3, plus: [0, 1, 0], minus: [3, 0, 0], axis: 'y', line: 0.8, plus_means: 'down', minus_means: 'up'}},
    {id: 'top', name: 'top', job_id: null, axis: 'y', start_offset_s: 0, plus: {top: -1}, minus: {top: 1}, meaning: '', gate: null},
  ],
});

describe('festival flow stock model', () => {
  it('conserves people moving between festivals and adds entrance flows', () => {
    const tl = buildTimeline(cfg());
    expect(tl.length).toBe(5);
    // t0: mid left 1 -> bottom +1, top -1 ; t1: mid right 2 -> top +2, bottom -2 ; t2: bottom cam enter 3 ; t3: exit 1
    expect(Array.from(tl.delta.top)).toEqual([-1, 1, 1, 1, 1]);
    expect(Array.from(tl.delta.bottom)).toEqual([1, -1, 2, 1, 1]);
    expect(tl.cameraActive.top).toBeNull();
  });

  it('uses the start count, clamps below zero with a flag, and extrapolates the recent rate', () => {
    const tl = buildTimeline(cfg());
    const z = zoneFlowAt(tl, 'bottom', 35, 2, 30);
    expect(z.raw).toBe(37);
    expect(z.rate).toBeCloseTo((2 - 1) / 2);
    expect(z.predicted).toBeCloseTo(37 + 0.5 * 30);
    const low = zoneFlowAt(tl, 'top', 0, 0, 0);
    expect(low.shown).toBe(0);
    expect(low.belowZero).toBe(true);
  });

  it('reports camera rates only while that camera has video', () => {
    expect(cameraRates(cfg().cameras[1], 1)).toBeNull();
    expect(cameraRates(cfg().cameras[1], 2)).toEqual({plus: 0, minus: 180});
  });
});

import {congestionLevel} from '../src/services/festivalFlow';

describe('congestion levels', () => {
  it('green below 50%, orange 50-80%, red from 80% of the reference', () => {
    expect(congestionLevel(49, 100).label).toBe('여유');
    expect(congestionLevel(50, 100).label).toBe('주의');
    expect(congestionLevel(79, 100).label).toBe('주의');
    expect(congestionLevel(80, 100).label).toBe('위험');
    expect(congestionLevel(500, 100).color).toBe('#dc2626');
    expect(congestionLevel(10, 0).label).toBe('여유');
  });
});

import {startCounts} from '../src/services/festivalFlow';

describe('automatic start counts', () => {
  it('keeps the default when flows never go below it, raises it otherwise', () => {
    const c = cfg();
    c.zones = {top: {initial: 35, auto_min_margin: 10}, bottom: {initial: 0, auto_min_margin: 10}};
    const s = startCounts(c, buildTimeline(c));
    expect(s.top).toMatchObject({count: 35, auto: false, minDelta: -1});
    expect(s.bottom).toMatchObject({count: 11, auto: true, minDelta: -1, minAt: 1});
  });
  it('without a margin the configured start is used as-is', () => {
    const c = cfg();
    expect(startCounts(c, buildTimeline(c)).bottom.count).toBe(35);
  });
});
