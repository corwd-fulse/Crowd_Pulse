import {beforeEach,expect,it,vi} from 'vitest';
import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import {MemoryRouter} from 'react-router-dom';
import CaptureSessions from '../src/pages/CaptureSessions';
import {api} from '../src/services/api';

vi.mock('../src/services/api',()=>({api:{get:vi.fn(),post:vi.fn(),defaults:{baseURL:'http://localhost:8000'}}}));
const venue={venue_id:'v-a',revision:2,name:'Public facility',address:'',venue_type:'facility'};
const session={session_id:'s-a',revision:3,venue_revision:1,data:{name:'Normal day',start:'2026-11-17T10:00:00+09:00',end:'2026-11-17T11:00:00+09:00',timing_basis:'planned',mode:'independent',is_holiday:null,holiday_evidence:'',is_event:false,event_name:'',event_phase:'normal',lighting_condition:'unknown',environment:'',clips:[]}};
beforeEach(()=>{
  vi.clearAllMocks();
  vi.mocked(api.get).mockImplementation(async path=>({data:String(path).endsWith('/sources')?[]:String(path).endsWith('/sessions')?[session]:[venue]}));
  vi.mocked(api.post).mockResolvedValue({data:session});
});
async function open(){render(<MemoryRouter><CaptureSessions/></MemoryRouter>);await screen.findByRole('option',{name:'Public facility'});fireEvent.change(screen.getByLabelText('관제 장소 선택'),{target:{value:'v-a'}});await screen.findByRole('option',{name:'Normal day · v3'});}

it('selects venue-scoped sessions and preserves the pinned map when revising metadata',async()=>{
  await open();
  fireEvent.change(screen.getByLabelText('촬영 세션 선택'),{target:{value:'s-a'}});
  fireEvent.change(screen.getByLabelText('세션 이름'),{target:{value:'Updated normal day'}});
  expect(screen.getByRole('button',{name:'학습용 CSV 다운로드'}).hasAttribute('disabled')).toBe(true);
  fireEvent.click(screen.getByRole('button',{name:'촬영 세션 저장'}));
  await waitFor(()=>expect(api.post).toHaveBeenCalledWith('/api/capture/venues/v-a/sessions/s-a/revisions',expect.objectContaining({base_revision:3,venue_revision:1,data:expect.objectContaining({name:'Updated normal day',is_event:false,event_phase:'normal'})})));
  expect(screen.getByRole('link',{name:'이 장소의 관제 지도 열기'}).getAttribute('href')).toBe('/dashboard?venue=v-a');
});

it('does not carry session clips into another selected venue',async()=>{
  await open();
  fireEvent.change(screen.getByLabelText('촬영 세션 선택'),{target:{value:'s-a'}});
  expect((screen.getByLabelText('세션 이름') as HTMLInputElement).value).toBe('Normal day');
  fireEvent.change(screen.getByLabelText('관제 장소 선택'),{target:{value:''}});
  expect(screen.queryByRole('button',{name:'학습용 CSV 다운로드'})).toBeNull();
});

it('registers cameras without a completed analysis or assumed capture clock',async()=>{
  await open();
  fireEvent.click(screen.getByRole('button',{name:'기존 영상 연결'}));
  fireEvent.click(screen.getByRole('button',{name:'기존 영상 연결'}));
  expect(screen.getAllByLabelText('카메라 ID')).toHaveLength(2);
  expect(screen.getAllByLabelText('촬영 시각 확인').every(input=>!(input as HTMLInputElement).checked)).toBe(true);
  expect(screen.getByRole('button',{name:'학습용 CSV 다운로드'}).hasAttribute('disabled')).toBe(true);
});

it('uploads multiple files sequentially and keeps them unbound until explicit analysis selection',async()=>{
  vi.mocked(api.get).mockImplementation(async path=>({data:String(path).endsWith('/upload-policy')?{max_bytes:1024,extensions:['.mp4']}:String(path).endsWith('/sources')?[]:String(path).endsWith('/sessions')?[session]:[venue]}));
  vi.mocked(api.post).mockResolvedValueOnce({data:{video_id:'video-a'}}).mockResolvedValueOnce({data:{video_id:'video-b'}});
  await open();
  fireEvent.change(screen.getByLabelText('여러 영상 업로드'),{target:{files:[new File(['one'],'one.mp4',{type:'video/mp4'}),new File(['two'],'two.mp4',{type:'video/mp4'})]}});
  await waitFor(()=>expect(screen.getAllByLabelText('카메라 ID')).toHaveLength(2));
  expect(api.post).toHaveBeenCalledTimes(2);
  expect(vi.mocked(api.post).mock.calls.map(call=>(call[2] as {params:{filename:string}}).params.filename)).toEqual(['one.mp4','two.mp4']);
  expect(screen.getAllByLabelText('촬영 시각 확인').every(input=>!(input as HTMLInputElement).checked)).toBe(true);
});
