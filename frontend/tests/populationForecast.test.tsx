import {beforeEach,expect,it,vi} from 'vitest';
import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import PopulationForecast from '../src/components/dashboard/PopulationForecast';
import {getForecastCatalog,uploadPopulationHistory} from '../src/services/populationForecast';
import type {PopulationResult} from '../src/services/populationForecast';
import sample from './fixtures/populationModelExecution.json';
vi.mock('../src/services/populationForecast',()=>({getForecastCatalog:vi.fn(),uploadPopulationHistory:vi.fn()}));
beforeEach(()=>{
  vi.clearAllMocks();
  vi.mocked(getForecastCatalog).mockResolvedValue({status:'ready',code:'ready',message:'ready',areas:[sample.items[0].area_id],models:{lightgbm:'lightgbm-1h-v1',lstm:'lstm-1h-36ep-seed0'},model_state:'not_loaded'});
  vi.mocked(uploadPopulationHistory).mockResolvedValue(sample as PopulationResult);
});
it('renders both frozen-model results from the recorded synthetic execution without replacing missing values',async()=>{
  render(<PopulationForecast/>);
  await screen.findByText(/아직 로드하지 않음/);
  fireEvent.change(screen.getByLabelText('생활인구 이력 파일'),{target:{files:[new File(['fixture'],'SYNTHETIC_interface_only.csv')]}});
  fireEvent.click(screen.getByRole('button',{name:'1시간 후 생활인구 예측'}));
  await waitFor(()=>expect(uploadPopulationHistory).toHaveBeenCalled());
  const light=sample.items.find(i=>i.model==='lightgbm')!;
  await screen.findByText(`${light.value!.toLocaleString('ko-KR',{maximumFractionDigits:1})}명`);
  fireEvent.change(screen.getByLabelText('생활인구 예측 모델'),{target:{value:'lstm'}});
  const lstm=sample.items.find(i=>i.model==='lstm')!;
  expect(screen.getByText(`${lstm.value!.toLocaleString('ko-KR',{maximumFractionDigits:1})}명`)).toBeTruthy();
});
it('disables prediction for an unavailable runtime and displays no invented values',async()=>{
  vi.mocked(getForecastCatalog).mockResolvedValue({status:'unavailable',code:'dependency_unavailable',message:'환경 미설치',areas:[],models:{},missing_dependencies:['torch']});
  render(<PopulationForecast/>);
  await screen.findByText('환경 미설치');
  expect(screen.getByRole('button',{name:'1시간 후 생활인구 예측'}).hasAttribute('disabled')).toBe(true);
  expect(document.querySelector('[data-forecast-status]')).toBeNull();
});
it('shows an execution error and clears model output',async()=>{
  vi.mocked(uploadPopulationHistory).mockRejectedValue(new Error('private worker failed'));
  render(<PopulationForecast/>);
  await screen.findByText(/아직 로드하지 않음/);
  fireEvent.change(screen.getByLabelText('생활인구 이력 파일'),{target:{files:[new File(['fixture'],'SYNTHETIC_interface_only.csv')]}});
  fireEvent.click(screen.getByRole('button',{name:'1시간 후 생활인구 예측'}));
  await screen.findByRole('alert');
  expect(document.querySelector('[data-forecast-status]')).toBeNull();
});
