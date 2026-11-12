import {act,fireEvent,render,screen,waitFor} from '@testing-library/react';
import {afterEach,beforeEach,expect,it,vi} from 'vitest';
import VideoRoiEditor from '../src/components/dashboard/VideoRoiEditor';
import {getVideoRoi,registerVideoRoi} from '../src/services/api';
import type {RoiSnapshot} from '../src/services/api';
vi.mock('../src/services/api',async original=>({...await original<typeof import('../src/services/api')>(),getVideoRoi:vi.fn(),registerVideoRoi:vi.fn()}));
const snapshot:RoiSnapshot={status:'approved',approved_for_analysis:true,video_resolution:[1920,1080],zones:[{name:'saved',polygon:[[10,10],[100,10],[100,100],[10,100]]}]};
beforeEach(()=>{vi.mocked(getVideoRoi).mockReset().mockResolvedValue(null);vi.mocked(registerVideoRoi).mockReset();});
afterEach(()=>vi.unstubAllGlobals());
async function mount() {
  const ready=vi.fn();const view=render(<VideoRoiEditor videoId="video-id" onReadiness={ready}/>);
  await waitFor(()=>expect(getVideoRoi).toHaveBeenCalled());
  await waitFor(()=>expect(screen.queryByText('ROI를 불러오는 중…')).toBeNull());
  const video=screen.getByLabelText('ROI 원본 영상');
  Object.defineProperties(video,{videoWidth:{value:1920},videoHeight:{value:1080}});fireEvent.loadedMetadata(video);
  return {...view,ready};
}
it('draws in source coordinates, previews and saves only a candidate',async()=>{
  vi.mocked(registerVideoRoi).mockImplementation(async(video_id,roi_snapshot)=>({video_id,roi_snapshot}));
  const {ready}=await mount();
  fireEvent.change(screen.getByLabelText('새 구역 이름'),{target:{value:'walkway'}});
  const svg=screen.getByRole('img',{name:'ROI 그리기 영역'});
  vi.spyOn(svg,'getBoundingClientRect').mockReturnValue({left:0,top:0,width:960,height:540} as DOMRect);
  for(const [clientX,clientY] of [[50,50],[200,50],[200,200],[50,200]])fireEvent.click(svg,{clientX,clientY,detail:1});
  fireEvent.click(screen.getByRole('button',{name:/폴리곤 완료/}));
  expect((screen.getByRole('button',{name:'후보 ROI 최초 저장'}) as HTMLButtonElement).disabled).toBe(true);
  fireEvent.click(screen.getByRole('button',{name:'저장 전 미리보기'}));
  fireEvent.click(screen.getByRole('button',{name:'후보 ROI 최초 저장'}));
  await screen.findByText(/후보 ROI를 저장했습니다/);
  expect(registerVideoRoi).toHaveBeenCalledWith('video-id',expect.objectContaining({status:'candidate',approved_for_analysis:false,zones:[{name:'walkway',polygon:[[100,100],[400,100],[400,400],[100,400]]}]}));
  expect(ready).toHaveBeenLastCalledWith({videoId:'video-id',production:false,test:true});
});
it('edits and deletes existing ROI only in a draft and blocks analysis while dirty',async()=>{
  vi.mocked(getVideoRoi).mockResolvedValue({video_id:'video-id',roi_snapshot:snapshot});
  const {ready}=await mount();
  await waitFor(()=>expect(ready).toHaveBeenLastCalledWith({videoId:'video-id',production:true,test:true}));
  fireEvent.change(screen.getByLabelText('구역 1 이름'),{target:{value:'modified'}});
  expect(ready).toHaveBeenLastCalledWith({videoId:'video-id',production:false,test:false});
  expect((screen.getByRole('button',{name:'후보 ROI 최초 저장'}) as HTMLButtonElement).disabled).toBe(true);
  fireEvent.click(screen.getByRole('button',{name:'구역 1 삭제'}));
  expect(screen.queryByLabelText('구역 1 이름')).toBeNull();expect(registerVideoRoi).not.toHaveBeenCalled();
  expect(snapshot.zones[0].name).toBe('saved');
});
it('does not treat a failed lookup as permission to overwrite',async()=>{
  vi.mocked(getVideoRoi).mockRejectedValue(new Error('offline'));
  render(<VideoRoiEditor videoId="video-id" onReadiness={()=>{}}/>);
  await screen.findByText('ROI 조회에 실패했습니다. 다시 조회하세요.');
  expect((screen.getByRole('button',{name:'후보 ROI 최초 저장'}) as HTMLButtonElement).disabled).toBe(true);
});

it('keeps drawing when a point is clicked inside an existing polygon and validates overlap on finish',async()=>{
  vi.mocked(getVideoRoi).mockResolvedValue({video_id:'video-id',roi_snapshot:snapshot});
  const {container}=await mount();
  const svg=screen.getByRole('img',{name:'ROI 그리기 영역'});
  vi.spyOn(svg,'getBoundingClientRect').mockReturnValue({left:0,top:0,width:1920,height:1080} as DOMRect);
  fireEvent.click(screen.getByRole('button',{name:'새 구역 그리기'}));
  fireEvent.click(container.querySelector('polygon')!,{clientX:20,clientY:20,detail:1});
  fireEvent.click(container.querySelector('polygon')!,{clientX:80,clientY:20,detail:1});
  fireEvent.click(container.querySelector('polygon')!,{clientX:50,clientY:80,detail:1});
  expect(screen.getByRole('button',{name:'폴리곤 완료 (3점)'})).toBeTruthy();
  fireEvent.click(screen.getByRole('button',{name:/폴리곤 완료/}));
  expect(screen.getByRole('alert').textContent).toContain('겹치');
  expect(screen.queryByLabelText('구역 2 이름')).toBeNull();
});

it('shows validation next to drawing controls and clears stale errors on edit, undo and cancel',async()=>{
  await mount();
  const svg=screen.getByRole('img',{name:'ROI 그리기 영역'});
  vi.spyOn(svg,'getBoundingClientRect').mockReturnValue({left:0,top:0,width:1920,height:1080} as DOMRect);
  fireEvent.change(screen.getByLabelText('새 구역 이름'),{target:{value:'line'}});
  for(const x of [10,20,30])fireEvent.click(svg,{clientX:x,clientY:10,detail:1});
  fireEvent.click(screen.getByRole('button',{name:/폴리곤 완료/}));
  const alert=screen.getByRole('alert');
  expect(alert.textContent).toContain('면적');
  expect(screen.getByRole('button',{name:/폴리곤 완료/}).parentElement?.nextElementSibling).toBe(alert);
  expect(document.activeElement).toBe(alert);
  fireEvent.click(screen.getByRole('button',{name:'마지막 점 취소'}));
  expect(screen.queryByRole('alert')).toBeNull();
  fireEvent.click(svg,{clientX:30,clientY:10,detail:1});
  fireEvent.click(screen.getByRole('button',{name:/폴리곤 완료/}));
  fireEvent.change(screen.getByLabelText('새 구역 이름'),{target:{value:'renamed'}});
  expect(screen.queryByRole('alert')).toBeNull();
  fireEvent.click(screen.getByRole('button',{name:/폴리곤 완료/}));
  fireEvent.click(screen.getByRole('button',{name:'그리기 취소'}));
  expect(screen.queryByRole('alert')).toBeNull();
});

it('removes stale polygons and approval labels when reloading fails, then recovers on retry',async()=>{
  vi.mocked(getVideoRoi).mockResolvedValueOnce({video_id:'video-id',roi_snapshot:snapshot}).mockRejectedValueOnce(new Error('503'));
  const {container,ready}=await mount();
  expect(container.querySelectorAll('polygon')).toHaveLength(1);
  fireEvent.click(screen.getByRole('button',{name:'ROI 다시 조회'}));
  await screen.findByText('ROI 조회에 실패했습니다. 다시 조회하세요.');
  expect(container.querySelectorAll('polygon')).toHaveLength(0);
  expect(screen.queryByLabelText('구역 1 이름')).toBeNull();
  expect(screen.getByText(/ROI 조회 실패 — 편집 불가/)).toBeTruthy();
  expect(ready).toHaveBeenLastCalledWith({videoId:'video-id',production:false,test:false});
  vi.mocked(getVideoRoi).mockResolvedValueOnce({video_id:'video-id',roi_snapshot:snapshot});
  fireEvent.click(screen.getByRole('button',{name:'ROI 다시 조회'}));
  await screen.findByLabelText('구역 1 이름');
  expect(container.querySelectorAll('polygon')).toHaveLength(1);
});

it('maintains 44 CSS pixel vertex targets across surface sizes without changing source coordinates',async()=>{
  let resized!: (entries: {contentRect:{width:number}}[])=>void;
  vi.stubGlobal('ResizeObserver',class {constructor(callback:typeof resized){resized=callback;} observe(){} disconnect(){}});
  vi.mocked(getVideoRoi).mockResolvedValue({video_id:'video-id',roi_snapshot:snapshot});
  const {container}=await mount();
  for(const width of [320,960]) {
    act(()=>resized([{contentRect:{width}}]));
    const handle=container.querySelector('circle[aria-label]')!;
    expect(Number(handle.getAttribute('r'))*2*width/1920).toBeCloseTo(44);
    expect(handle.getAttribute('cx')).toBe('10');
    expect(handle.getAttribute('cy')).toBe('10');
  }
});

it('drags the nearest vertex when enlarged touch targets overlap',async()=>{
  vi.stubGlobal('PointerEvent',MouseEvent);
  vi.mocked(getVideoRoi).mockResolvedValue({video_id:'video-id',roi_snapshot:snapshot});
  const {container}=await mount();
  const svg=screen.getByRole('img',{name:'ROI 그리기 영역'});
  Object.assign(svg,{setPointerCapture:vi.fn()});
  vi.spyOn(svg,'getBoundingClientRect').mockReturnValue({left:0,top:0,width:320,height:180} as DOMRect);
  // A nearby handle painted later may receive the hit, but choose the closest vertex.
  fireEvent.pointerDown(container.querySelectorAll('circle[aria-label]')[3],{clientX:10/6,clientY:10/6});
  fireEvent.pointerMove(svg,{clientX:20/6,clientY:20/6});
  const draft=JSON.parse(container.querySelector('details pre')!.textContent!);
  expect(draft.zones[0].polygon[0]).toEqual([20,20]);
  expect(draft.zones[0].polygon[3]).toEqual([10,100]);
});
