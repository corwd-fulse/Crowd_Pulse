"""Optional CPU-only playback copies; the analysis source is always unchanged."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from decimal import Decimal
import hashlib,json,os,shutil,subprocess,tempfile,threading

ROOT=Path(__file__).resolve().parents[3]
CACHE=ROOT/'backend/uploads/playback'
_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='playback-copy')
_lock=threading.Lock();_tasks={}

def binary(name):
    configured=os.getenv('CROWD_'+name.upper()+'_BINARY')
    if configured:
        path=Path(configured)
        if not path.is_file():raise ValueError(f'{name} executable not found')
        return str(path)
    installed=shutil.which(name)
    if installed:return installed
    paths=sorted((ROOT/'.tools/ffmpeg').glob(f'*/bin/{name}.exe'))
    if paths:return str(paths[-1])
    raise ValueError('FFmpeg/ffprobe unavailable; configure CROWD_FFMPEG_BINARY and CROWD_FFPROBE_BINARY')

def execute(args,**kwargs):
    return subprocess.run(args,check=True,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,**kwargs)

def key(source):
    s=source.stat()
    return hashlib.sha256(f'{source.resolve()}:{s.st_size}:{s.st_mtime_ns}:h264-cfr-v1'.encode()).hexdigest()

@lru_cache(maxsize=64)
def probe_cached(path,size,mtime):
    raw=execute([binary('ffprobe'),'-v','error','-select_streams','v:0','-show_entries',
                 'stream=codec_name,codec_tag_string,pix_fmt,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,start_time',
                 '-of','json',path],capture_output=True,timeout=30)
    streams=json.loads(raw.stdout)['streams']
    if not streams:raise ValueError('No video stream')
    info=streams[0]
    info['browser_candidate']=info.get('codec_name') in ('h264','vp8','vp9','av1')
    return info

def probe(source):
    s=source.stat();return dict(probe_cached(str(source),s.st_size,s.st_mtime_ns))

def status(source):
    identity=key(source);manifest=CACHE/(identity+'.json');target=CACHE/(identity+'.mp4')
    if manifest.is_file() and target.is_file():return {**json.loads(manifest.read_text()),'status':'ready'}
    with _lock:
        future=_tasks.get(identity)
        if future:
            if not future.done():return dict(status='processing')
            if future.exception():return dict(status='failed',error=str(future.exception())[:500])
    return dict(status='not_prepared')

def request(source):
    current=status(source)
    if current['status'] in ('ready','processing'):return current
    binary('ffmpeg');binary('ffprobe');CACHE.mkdir(parents=True,exist_ok=True)
    with _lock:
        identity=key(source)
        # One conversion per API process; no waiting videos retaining resources.
        if any(not f.done() for f in _tasks.values()):raise ValueError('Playback converter is busy; retry later')
        for k in list(_tasks):
            if _tasks[k].done():_tasks.pop(k)
        _tasks[identity]=_executor.submit(convert,source,CACHE/(identity+'.mp4'))
    return dict(status='processing')

def timestamps(source,destination):
    with destination.open('wb') as stream:
        execute([binary('ffprobe'),'-v','error','-select_streams','v:0','-show_frames','-show_entries',
                 'frame=best_effort_timestamp_time','-of','csv=p=0',str(source)],stdout=stream,stderr=subprocess.PIPE,timeout=180)
    if destination.stat().st_size>64*1024**2:raise ValueError('Frame timestamp validation exceeds disk budget')
    def values():
        with destination.open() as stream:
            for line in stream:
                text=line.strip().split(',')[0]
                if text:
                    try:yield Decimal(text)
                    except Exception:raise ValueError('Missing frame timestamp') from None
    return values()

def verify(source,target,folder):
    from itertools import zip_longest
    info=probe(source);out=probe(target)
    if out['codec_name']!='h264' or (info['width'],info['height'])!=(out['width'],out['height']):raise ValueError('Playback codec/resolution mismatch')
    num,den=map(Decimal,info['avg_frame_rate'].split('/'));fps=num/den
    if fps<=0:raise ValueError('Invalid source fps')
    a=timestamps(source,folder/'original.pts');b=timestamps(target,folder/'playback.pts')
    count=0;first_a=first_b=None;maximum=Decimal(0)
    for left,right in zip_longest(a,b):
        if left is None or right is None:raise ValueError('Playback frame count mismatch')
        if first_a is None:
            first_a,first_b=left,right
            if abs(right)>Decimal('.00002'):raise ValueError('Playback does not start at frame-zero time; exact replay alignment unavailable')
        relative_a,relative_b=left-first_a,right-first_b
        if abs(relative_a-Decimal(count)/fps)>Decimal('.00002'):raise ValueError('Variable/nonuniform timestamps: exact frame-index replay requires CFR source; original preserved')
        error=abs(relative_a-relative_b);maximum=max(maximum,error)
        if error>Decimal('.00002'):raise ValueError('Playback frame timeline mismatch')
        count+=1
    if not count:raise ValueError('No decoded frames')
    return dict(codec='h264',frame_count=count,fps=float(fps),max_frame_timestamp_error_seconds=float(maximum),verified_every_frame=True,analysis_uses_original=True)

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()

def convert(source,target):
    source=Path(source);target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    before=key(source);size=source.stat().st_size
    if size>8*1024**3 or shutil.disk_usage(target.parent).free<size*4+1024**3:raise ValueError('Insufficient conversion disk reserve or source exceeds 8 GiB')
    info=probe(source)
    if info['width']%2 or info['height']%2:raise ValueError('Odd source dimensions: cannot preserve resolution in yuv420p')
    from .artifact_lock import acquire
    with acquire(target.with_suffix('.lock')):
        with tempfile.TemporaryDirectory(prefix='playback-',dir=target.parent) as folder:
            folder=Path(folder);temporary=folder/'copy.mp4'
            execute([binary('ffmpeg'),'-hide_banner','-loglevel','error','-nostdin','-n','-threads','2','-copyts','-start_at_zero','-i',str(source),
                '-map','0:v:0','-map','0:a?','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-threads','2',
                '-fps_mode','passthrough','-c:a','aac','-movflags','+faststart','-fs',str(16*1024**3),str(temporary)],capture_output=True,timeout=3600)
            result=verify(source,temporary,folder)
            if key(source)!=before:raise ValueError('Source changed during playback conversion')
            result.update(source_sha256=sha(source),playback_sha256=sha(temporary),bytes=temporary.stat().st_size)
            manifest=folder/'manifest.json';manifest.write_text(json.dumps(result),encoding='utf-8')
            os.replace(temporary,target);os.replace(manifest,target.with_suffix('.json'))
            return result
