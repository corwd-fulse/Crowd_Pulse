import {act, fireEvent, render, screen, waitFor} from '@testing-library/react';
import {beforeEach, describe, expect, it, vi} from 'vitest';
import type {ReactElement} from 'react';
import AnalysisResults from '../src/components/dashboard/AnalysisResults';
import {getVisionResultBundle} from '../src/services/api';
import type {AnalysisJobResponse, VisionResultBundle} from '../src/services/api';
import {crowdSeries, directionLabel} from '../src/services/visionPresentation';
import bundleJson from './fixtures/visionBundle.json';
import jobJson from './fixtures/analysisJob.json';

vi.mock('../src/services/api', async (importOriginal) => ({
  ...await importOriginal<typeof import('../src/services/api')>(), getVisionResultBundle: vi.fn(),
}));
vi.mock('recharts', async (importOriginal) => {
  const react = await import('react');
  return {...await importOriginal<typeof import('recharts')>(),
    ResponsiveContainer: ({children}: {children: ReactElement<{width: number; height: number}>}) => react.cloneElement(children, {width:800,height:240}),
  };
});

const fixture = bundleJson as VisionResultBundle;
const job = jobJson as AnalysisJobResponse;
const load = vi.mocked(getVisionResultBundle);
beforeEach(() => {load.mockReset(); load.mockResolvedValue(structuredClone(fixture));});

describe('real-result presentation', () => {
  it('shows FAST completion and explicit unanalysed flow/bottleneck while displaying individual estimates', async () => {
    const fast=structuredClone(fixture);
    fast.summary.analysis_profile='fast';fast.scores=[];
    fast.metrics=fast.metrics.map(row=>({...row,has_flow:false,flow_angle_deg:null,magnitude_px:null,motion:null,low_confidence:null}));
    load.mockResolvedValue(fast);
    render(<AnalysisResults job={{...job,analysis_profile:'fast'}}/>);
    await screen.findByText('FAST 빠른 분석 완료');
    expect(screen.getByText(/위험 없음 판정이 아닙니다/)).toBeTruthy();
    expect(screen.getAllByText('미분석 (FAST)').length).toBeGreaterThan(0);
    expect(screen.queryByText(/빈 결과도 정상 분석 완료/)).toBeNull();
    expect(screen.getAllByText(/° \(미검증 방향값\)/).length).toBeGreaterThan(0);
  });
  it('fetches completed results, labels test/candidate/unvalidated, preserves decimal means', async () => {
    const {container} = render(<AnalysisResults job={job}/>);
    expect(screen.getByRole('status').textContent).toContain('불러오는 중');
    await screen.findByText('테스트 분석 결과');
    expect(load).toHaveBeenCalledWith(job.id, expect.any(AbortSignal));
    expect(screen.getByText('미승인 ROI')).toBeTruthy();
    expect(screen.getByText('방향 품질 미검증')).toBeTruthy();
    expect(screen.getAllByText('3.9').length).toBeGreaterThan(0);
    expect(container.querySelectorAll('.recharts-line-curve')).toHaveLength(2);
    expect(container.querySelectorAll('tbody')[0].textContent).not.toContain('305.214');
    expect(screen.getAllByText('미검증 방향값').length).toBeGreaterThan(0);
    expect(screen.getAllByText('신뢰 낮음').length).toBeGreaterThan(0);
    expect(screen.getByRole('link', {name: /individual_directions.*5,280/})).toBeTruthy();
  });

  it('gates result requests until succeeded and exposes failure messages', () => {
    const {rerender} = render(<AnalysisResults job={{...job,status:'running'}}/>);
    expect(load).not.toHaveBeenCalled();
    expect(screen.getByRole('status').textContent).toContain('분석 진행 중');
    rerender(<AnalysisResults job={{...job,status:'failed',error_message:'ROI_NOT_APPROVED'}}/>);
    expect(screen.getByRole('alert').textContent).toContain('ROI_NOT_APPROVED');
    expect(load).not.toHaveBeenCalled();
  });

  it('distinguishes empty results from failures and never falls back to mock data', async () => {
    load.mockResolvedValue({...fixture,metrics:[],scores:[]});
    const {container} = render(<AnalysisResults job={job}/>);
    await screen.findByText('저장된 구역 지표가 없습니다.');
    expect(screen.getByText(/빈 결과도 정상 분석 완료/)).toBeTruthy();
    expect(screen.queryByRole('alert')).toBeNull();
    expect(container.textContent).not.toContain('09:00');
    expect(container.querySelector('[data-source="mock"]')).toBeNull();
  });

  it('shows a retryable API error instead of mock results', async () => {
    load.mockRejectedValueOnce(new Error('offline'));
    render(<AnalysisResults job={job}/>);
    await screen.findByRole('alert');
    fireEvent.click(screen.getByRole('button',{name:'결과 다시 조회'}));
    await screen.findByText('테스트 분석 결과');
    expect(load).toHaveBeenCalledTimes(2);
  });

  it('discards a late response after the selected job changes', async () => {
    let resolveOld!: (data: VisionResultBundle) => void;
    load.mockImplementationOnce(() => new Promise((resolve) => {resolveOld=resolve;}));
    const {rerender} = render(<AnalysisResults job={job}/>);
    const other = {...job,id:'other-job'};
    load.mockResolvedValueOnce({...fixture,summary:{...fixture.summary,analysis_job_id:other.id}});
    rerender(<AnalysisResults job={other}/>);
    await screen.findByText(/작업 other-job/);
    await act(async () => resolveOld(fixture));
    expect(screen.queryByText(new RegExp(`작업 ${job.id}`))).toBeNull();
  });

  it('filters zones and paginates long metric tables', async () => {
    render(<AnalysisResults job={job}/>);
    await screen.findByText('테스트 분석 결과');
    fireEvent.click(screen.getByRole('button',{name:'다음 지표'}));
    expect(screen.getByText(/2 \/ 2쪽/)).toBeTruthy();
    fireEvent.change(screen.getByRole('combobox',{name:'결과 구역'}),{target:{value:'walkway_near'}});
    expect(screen.getByText(/1 \/ 1쪽/)).toBeTruthy();
    await waitFor(() => expect(screen.queryByText(/2 \/ 2쪽/)).toBeNull());
  });
});

describe('semantic safeguards', () => {
  it('preserves zero and null and uses relative milliseconds for chart points', () => {
    const row = fixture.metrics[0];
    const data = crowdSeries([{...row,mean_person_count:0}, {...row,video_offset_ms:1000,has_observation:false,mean_person_count:null}]);
    expect(data.points[0].zone0).toBe(0);
    expect(data.points[1].zone0).toBeNull();
    expect(data.points[1].video_offset_ms).toBe(1000);
  });

  it('requires both global and per-row confirmation and suppresses low-confidence/null directions', () => {
    const summary = structuredClone(fixture.summary);
    const row = {...fixture.metrics[0],flow_direction_confirmed:true};
    expect(directionLabel(summary,row,'flow')).toBe('미검증 방향값');
    summary.roi_status='approved'; summary.roi_snapshot.approved_for_analysis=true;
    summary.direction_presentation={...summary.direction_presentation,quality:'validated',confirmed_for_display:true};
    expect(directionLabel(summary,row,'flow')).toContain('검증된 추정값');
    expect(directionLabel(summary,{...row,low_confidence:true},'flow')).toBe('신뢰 낮음');
    expect(directionLabel(summary,{...row,flow_angle_deg:null},'flow')).toBe('데이터 없음');
    expect(directionLabel(summary,{...row,flow_direction_confirmed:false},'flow')).toBe('미검증 방향값');
  });
});
