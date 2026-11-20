// Shared, byte/count-bounded page cache. Eviction aborts its own pending request.
type Entry={promise:Promise<unknown>;controller:AbortController;bytes:number};
const pages=new Map<string,Entry>();const MAX_PAGES=12,MAX_BYTES=24*1024*1024;
function trim(){let total=[...pages.values()].reduce((n,e)=>n+e.bytes,0);while(pages.size>MAX_PAGES||total>MAX_BYTES){const key=pages.keys().next().value!;const e=pages.get(key)!;total-=e.bytes;e.controller.abort();pages.delete(key);}}
export function cachedWindow<T>(key:string,load:(signal:AbortSignal)=>Promise<T>):Promise<T>{
  const old=pages.get(key);if(old){pages.delete(key);pages.set(key,old);return old.promise as Promise<T>;}
  const controller=new AbortController();const entry:Entry={controller,bytes:0,promise:Promise.resolve()};
  entry.promise=load(controller.signal).then(data=>{entry.bytes=JSON.stringify(data).length*2;trim();return data;}).catch(error=>{if(pages.get(key)===entry)pages.delete(key);throw error;});
  pages.set(key,entry);trim();return entry.promise as Promise<T>;
}
export async function pageReady<T>(load:()=>Promise<{status:number;data:T}>,signal:AbortSignal):Promise<T>{
  const deadline=Date.now()+300000;
  for(;;){if(signal.aborted)throw new DOMException('Aborted','AbortError');const r=await load();if(r.status!==202)return r.data;
    if(Date.now()>deadline)throw new Error('Replay index preparation timed out; retry later');
    await new Promise<void>((resolve,reject)=>{const abort=()=>{clearTimeout(timer);reject(new DOMException('Aborted','AbortError'));};const timer=setTimeout(()=>{signal.removeEventListener('abort',abort);resolve();},500);signal.addEventListener('abort',abort,{once:true});});
  }
}
