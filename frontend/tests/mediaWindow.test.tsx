import {renderHook,waitFor} from '@testing-library/react';
import {expect,it,vi} from 'vitest';
import {useMediaWindow} from '../src/hooks/useMediaWindow';
import {cachedWindow} from '../src/services/mediaWindowCache';
const page=(start:number)=>({frame_count:90000,fps:25,window:{start_ms:start,end_ms:start+5000,next_start_ms:start+5000}});

it('seeks directly to a distant interval, prefetches and swaps without accumulating old pages',async()=>{
  const fetch=vi.fn(async(_id:string,_signal?:AbortSignal,start=0)=>page(start));
  const {result,rerender}=renderHook(({t})=>useMediaWindow('job','video',t,fetch),{initialProps:{t:0}});
  await waitFor(()=>expect(result.current.data?.window?.start_ms).toBe(0));
  rerender({t:3.5});await waitFor(()=>expect(fetch).toHaveBeenCalledWith('job',expect.any(AbortSignal),5000,10000));
  rerender({t:5.2});await waitFor(()=>expect(result.current.data?.window?.start_ms).toBe(5000));
  rerender({t:1800});await waitFor(()=>expect(result.current.data?.window?.start_ms).toBe(1800000));
  rerender({t:3599});await waitFor(()=>expect(result.current.data?.window?.start_ms).toBe(3595000));
  expect(result.current.error).toBe(false);
});

it('ignores late responses from the previous camera/job',async()=>{
  let resolveOld:(v:ReturnType<typeof page>)=>void=()=>{};
  const fetch=vi.fn((id:string)=>id==='old'?new Promise<ReturnType<typeof page>>(resolve=>{resolveOld=resolve;}):Promise.resolve(page(0)));
  const {result,rerender}=renderHook(({id})=>useMediaWindow(id,'video',0,fetch),{initialProps:{id:'old'}});
  rerender({id:'new'});await waitFor(()=>expect(result.current.loaded).toBe(true));
  resolveOld(page(99000));await new Promise(resolve=>setTimeout(resolve,10));
  expect(result.current.data?.window?.start_ms).toBe(0);
});

it('seeks inside a shortened dense interval without walking every intervening page',async()=>{
  const fetch=vi.fn(async(_id:string,_signal?:AbortSignal,start=0)=>({
    ...page(start),window:{start_ms:start,end_ms:start+130.13,next_start_ms:start+130.13},
  }));
  const {result,rerender}=renderHook(({t})=>useMediaWindow('job','video',t,fetch),{initialProps:{t:0}});
  await waitFor(()=>expect(result.current.data?.window?.start_ms).toBe(0));
  rerender({t:3.25});
  await waitFor(()=>expect(result.current.data?.window?.start_ms).toBe(3250));
  expect(fetch.mock.calls.map(call=>call[2])).toEqual([0,130.13,3250,3380.13]);
});

it('evicts old byte-heavy cache entries and deduplicates shared requests',async()=>{
  const load=vi.fn(async()=>({payload:'x'.repeat(5*1024*1024)}));
  await cachedWindow('large-test-a',load);await cachedWindow('large-test-a',load);
  expect(load).toHaveBeenCalledTimes(1);
  await cachedWindow('large-test-b',load);await cachedWindow('large-test-c',load);
  await cachedWindow('large-test-a',load);expect(load).toHaveBeenCalledTimes(4);
});
