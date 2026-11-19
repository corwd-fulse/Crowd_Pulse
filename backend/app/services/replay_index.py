"""Disk-backed, validated replay index. Raw observations are never sampled/deleted."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
import csv,hashlib,json,math,os,shutil,sqlite3,threading,time
from pathlib import Path
from . import detection_view
from .roi_geometry import inside
from .replay_state import TrackState
from .zone_replay import VERSION

INDEX_VERSION='sqlite-observed-v1'
MAX_SOURCE_BYTES=8*1024**3
MAX_SOURCE_ROWS=100_000_000
MAX_INDEX_BYTES=24*1024**3
MAX_RESPONSE_POINTS=50_000
MAX_RESPONSE_BYTES=12*1024**2
MAX_WINDOW_MS=30_000
DEFAULT_WINDOW_MS=5_000
_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='replay-index')
_pending={};_lock=threading.Lock()


class IndexPending(ValueError):pass
class WindowTooDense(ValueError):pass


class Connection(sqlite3.Connection):
    def __exit__(self,*args):
        try:return super().__exit__(*args)
        finally:self.close()


def connect(path,write=False):
    path=Path(path).resolve()
    db=sqlite3.connect(str(path) if write else path.as_uri()+'?mode=ro',uri=not write,timeout=30,factory=Connection)
    db.execute('PRAGMA cache_size=-16384');db.execute('PRAGMA temp_store=FILE')
    return db


def identity(job,result):
    path=detection_view.tracking_path(job,result);frames=path.with_name('video_frames.csv')
    a=path.stat();b=frames.stat() if frames.is_file() else None
    meta=dict(version=VERSION,index_version=INDEX_VERSION,analysis_job_id=str(job.id),video_id=str(job.video_id),
        fps=float(result.execution_snapshot['fps']),frame_count=int(result.execution_snapshot['frame_count']),
        video_resolution=[result.width,result.height],roi_snapshot=result.roi_snapshot)
    recorded=result.execution_snapshot.get('settings',{}).get('transitions')
    if recorded:
        if recorded.get('version')!=VERSION:raise ValueError('Recorded transition policy differs from replay')
        meta['recorded_event_digest']=event_digest(recorded.get('events',[]))
    if not math.isfinite(meta['fps']) or meta['fps']<=0 or not 0<meta['frame_count']<=10_000_000 or a.st_size>MAX_SOURCE_BYTES:
        raise ValueError('Indexed replay source budget exceeded')
    signature=json.dumps([meta,result.fingerprint,a.st_size,a.st_mtime_ns,b.st_size if b else None,b.st_mtime_ns if b else None],sort_keys=True)
    root=detection_view.storage.BASE_DIR/'uploads/replay-index';root.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(signature.encode()).hexdigest()
    return path,frames,meta,root/(key+'.sqlite'),(a.st_size,a.st_mtime_ns),((b.st_size,b.st_mtime_ns) if b else None)


def event_digest(events):
    digest=hashlib.sha256()
    for event in events:digest.update(json.dumps(event,sort_keys=True).encode());digest.update(b'\n')
    return digest.hexdigest()


def ensure(job,result,*,wait=False):
    args=identity(job,result);destination=args[3]
    if destination.is_file():return destination
    with _lock:
        future=_pending.get(str(destination))
        if future is None:
            # Completed entries do not retain index results or source payloads.
            for key in list(_pending):
                if _pending[key].done():_pending.pop(key)
            if len(_pending)>=4:raise IndexPending('Index queue is full; retry shortly')
            future=_executor.submit(build,*args);_pending[str(destination)]=future
    if wait or future.done():
        try:future.result()
        except Exception:
            with _lock:
                if _pending.get(str(destination)) is future:_pending.pop(str(destination),None)
            raise
        return destination
    raise IndexPending('Preparing validated disk index')


def build(path,frame_path,meta,destination,source_stat,frame_stat):
    from .artifact_lock import acquire
    with acquire(destination.with_suffix('.lock')):
        if destination.is_file():return
        return _build_locked(path,frame_path,meta,destination,source_stat,frame_stat)


def _build_locked(path,frame_path,meta,destination,source_stat,frame_stat):
    temporary=destination.with_suffix(f'.{os.getpid()}.building')
    if shutil.disk_usage(destination.parent).free<source_stat[0]*6+1024**3:
        raise ValueError('Insufficient replay index storage reserve')
    db=None
    try:
        db=connect(temporary,True)
        db.executescript('''CREATE TABLE boxes(frame INTEGER,t_ms INTEGER,tid INTEGER,x1 REAL,y1 REAL,x2 REAL,y2 REAL,confidence REAL);
            CREATE TABLE frames(frame INTEGER PRIMARY KEY,t_ms INTEGER,boxes TEXT,points TEXT,counts TEXT,observed INTEGER);
            CREATE TABLE events(frame INTEGER,tid INTEGER,kind TEXT,src TEXT,dst TEXT,payload TEXT);
            CREATE TABLE diagnostics(frame INTEGER,tid INTEGER,payload TEXT);
            CREATE TABLE od(src TEXT,dst TEXT,n INTEGER,PRIMARY KEY(src,dst));
            CREATE TABLE metadata(payload TEXT);''')
        width,height=meta['video_resolution'];fps=meta['fps'];count=meta['frame_count'];batch=[];rows=0
        with path.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream)
            if reader.fieldnames!=['frame','t_ms','track_id','x1','y1','x2','y2','conf']:raise ValueError('Invalid tracking header')
            for row in reader:
                f,t=int(row['frame']),int(row['t_ms']);tid=int(row['track_id']) if row['track_id'] else None
                x1,y1,x2,y2,confidence=[float(row[k]) for k in ('x1','y1','x2','y2','conf')]
                if not (0<=f<count and abs(t-int(f/fps*1000))<=1 and (tid is None or tid>=0) and
                    all(math.isfinite(n) for n in (x1,y1,x2,y2,confidence)) and 0<=x1<x2<=width and 0<=y1<y2<=height and 0<=confidence<=1):raise ValueError('Invalid tracking row')
                batch.append((f,t,tid,x1,y1,x2,y2,confidence));rows+=1
                if rows>MAX_SOURCE_ROWS:raise ValueError('Tracking index row budget exceeded')
                if len(batch)>=2000:db.executemany('INSERT INTO boxes VALUES(?,?,?,?,?,?,?,?)',batch);batch.clear()
            if batch:db.executemany('INSERT INTO boxes VALUES(?,?,?,?,?,?,?,?)',batch)
        db.execute('CREATE INDEX boxes_frame ON boxes(frame)');db.execute('CREATE INDEX boxes_track ON boxes(tid,frame)')
        zones=meta['roi_snapshot']['zones'];polygons={z['name']:z['polygon'] for z in zones}
        def diagnostic(d):db.execute('INSERT INTO diagnostics VALUES(?,?,?)',(d['frame'],d['track_id'],json.dumps(d)))
        def event(e):db.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',(e['frame'],e['track_id'],e['kind'],e['from_zone'],e['to_zone'],json.dumps(e)))
        def classify(p):
            matches=[z['name'] for z in zones if inside(p,z['polygon'])]
            return matches[0] if len(matches)==1 else '' if not matches else None
        for frame,group in groupby(db.execute('SELECT * FROM boxes ORDER BY frame,rowid'),key=lambda r:r[0]):
            boxes=[];points=[];counts={z['name']:0 for z in zones};ambiguous=False;t=None
            for f,tm,tid,x1,y1,x2,y2,confidence in group:
                if t is not None and t!=tm:raise ValueError('Inconsistent detection timestamps')
                t=tm;p=[(x1+x2)/2,y2];zone=classify(p)
                if zone in counts:counts[zone]+=1
                if zone is None:ambiguous=True
                boxes.append(dict(track_id=tid,x1=x1,y1=y1,x2=x2,y2=y2,confidence=confidence))
                points.append(dict(track_id=tid,x=p[0],y=p[1],zone=zone,confidence=confidence))
                if tid is None:diagnostic(dict(frame=f,track_id=None,reason='missing_track_id'))
                if len(boxes)>20_000:raise ValueError('Single frame exceeds memory budget; no observations were sampled')
            db.execute('INSERT INTO frames VALUES(?,?,?,?,?,0)',(frame,t,json.dumps(boxes),json.dumps(points),json.dumps(None if ambiguous else counts)))
            if frame%1000==0 and temporary.stat().st_size>MAX_INDEX_BYTES:raise ValueError('Replay disk budget exceeded')
        verified=0
        if frame_stat is not None:
            with frame_path.open(encoding='utf-8-sig',newline='') as stream:
                reader=csv.DictReader(stream)
                if reader.fieldnames!=['frame','t_ms','count']:raise ValueError('Invalid frame index header')
                for row in reader:
                    f,t,n=[int(row[k]) for k in ('frame','t_ms','count')]
                    actual=db.execute('SELECT observed,json_array_length(boxes) FROM frames WHERE frame=?',(f,)).fetchone()
                    if not 0<=f<count or abs(t-int(f/fps*1000))>1 or n<0 or actual and actual[0] or n!=(actual[1] if actual else 0):raise ValueError('Frame/detection mismatch')
                    if actual:db.execute('UPDATE frames SET observed=1 WHERE frame=?',(f,))
                    else:db.execute('INSERT INTO frames VALUES(?,?,?, ?,?,1)',(f,t,'[]','[]',json.dumps({z['name']:0 for z in zones})))
                    verified+=1
            if verified!=count:raise ValueError('Incomplete frame index')
        for tid,group in groupby(db.execute('SELECT frame,t_ms,tid,x1,y1,x2,y2,confidence FROM boxes WHERE tid IS NOT NULL ORDER BY tid,frame,rowid'),key=lambda r:r[2]):
            state=TrackState(tid,polygons,event,diagnostic)
            for f,t,_,x1,y1,x2,y2,c in group:
                p=[(x1+x2)/2,y2];state.step(f,t,p,classify(p),c)
            pair=state.finish(count)
            if pair:db.execute('INSERT INTO od VALUES(?,?,1) ON CONFLICT(src,dst) DO UPDATE SET n=n+1',pair)
        db.executescript("CREATE INDEX events_frame ON events(frame);CREATE INDEX events_tid ON events(tid);CREATE INDEX events_direction ON events(kind,src,dst,frame);CREATE INDEX diagnostics_frame ON diagnostics(frame);CREATE INDEX diagnostics_resumed ON diagnostics(coalesce(json_extract(payload,'$.resumed_frame'),frame));")
        if meta.get('recorded_event_digest') and meta['recorded_event_digest']!=event_digest(json.loads(r[0]) for r in db.execute('SELECT payload FROM events ORDER BY frame,tid,kind')):
            raise ValueError('Recorded transition events differ from indexed observations')
        meta.update(mode='observed_replay_not_prediction',policy=dict(min_dwell_frames=3,hysteresis_px=3.,min_displacement_px=15.,overlap='ambiguous_unanalyzed',gap='observed_outside',id_discontinuity='split_without_linking',visibility='display_only'),
            coverage=dict(verified_frame_count=verified,complete_frame_index=verified==count,
                incomplete_track_ids=db.execute("SELECT count(DISTINCT tid) FROM diagnostics WHERE json_extract(payload,'$.reason') IN ('id_gap_or_duplicate','suspected_id_switch','ambiguous_overlap')").fetchone()[0],
                unresolved_paths=db.execute('SELECT count(*) FROM diagnostics').fetchone()[0]),
            od_note='One start/end pair per unfragmented observed ID with zone change; censored observations, not true journey OD',
            first_uncertain_frame=db.execute('SELECT min(frame) FROM diagnostics').fetchone()[0])
        db.execute('INSERT INTO metadata VALUES(?)',(json.dumps(meta),));db.commit();db.close();db=None
        if (path.stat().st_size,path.stat().st_mtime_ns)!=source_stat or (frame_stat is not None and (frame_path.stat().st_size,frame_path.stat().st_mtime_ns)!=frame_stat):raise ValueError('Tracking source changed during indexing')
        if temporary.stat().st_size>MAX_INDEX_BYTES:raise ValueError('Replay disk budget exceeded')
        os.replace(temporary,destination)
    finally:
        if db is not None:db.close()
        # Only artifacts newly allocated by this builder, never CSVs or old indices.
        temporary.unlink(missing_ok=True)


def window(index,start_ms,end_ms,*,boxes_only=False):
    if start_ms<0 or end_ms<=start_ms or end_ms-start_ms>MAX_WINDOW_MS:raise ValueError('Replay window must be 0..30 seconds')
    with connect(index) as db:
        meta=json.loads(db.execute('SELECT payload FROM metadata').fetchone()[0]);fps=meta['fps']
        if start_ms>=meta['frame_count']/fps*1000:raise ValueError('Requested time is outside the source video')
        requested_frame=math.floor(start_ms/1000*fps+1e-6)
        first=max(0,requested_frame-1);last=min(meta['frame_count'],math.ceil(end_ms/1000*fps))
        frames=[];points=0;size=0
        for f,t,b,p,c,observed in db.execute('SELECT * FROM frames WHERE frame>=? AND frame<? ORDER BY frame',(first,last)):
            n=len(json.loads(b)) if boxes_only else len(json.loads(p));size+=len(b if boxes_only else p)+len(c)+100;points+=n
            if points>MAX_RESPONSE_POINTS or size>MAX_RESPONSE_BYTES:
                last=f;break
            frames.append(dict(frame=f,t_ms=t,boxes=json.loads(b),observed=bool(observed)) if boxes_only else dict(frame=f,t_ms=t,points=json.loads(p),_counts=json.loads(c),observed=bool(observed)))
        if last<=requested_frame:raise WindowTooDense('Requested frame and its context exceed response budget; no rows were sampled')
        actual_end=last/fps*1000
        result={k:v for k,v in meta.items() if k!='first_uncertain_frame'}
        result.update(window=dict(start_ms=first/fps*1000,end_ms=actual_end,requested_start_ms=start_ms,requested_end_ms=end_ms,
            next_start_ms=actual_end if last<meta['frame_count'] else None,max_points=MAX_RESPONSE_POINTS),frames=frames)
        if boxes_only:return {k:v for k,v in result.items() if k in ('analysis_job_id','video_id','fps','frame_count','video_resolution','frames','window','index_version')}
        # Cumulative counts are read from the disk index, not restarted at a page edge.
        incoming=Counter(dict(db.execute("SELECT dst,count(*) FROM events WHERE kind='boundary' AND dst<>'' AND frame<? GROUP BY dst",(first,))))
        outgoing=Counter(dict(db.execute("SELECT src,count(*) FROM events WHERE kind='boundary' AND src<>'' AND frame<? GROUP BY src",(first,))))
        event_first=max(0,first-math.ceil(fps*60))
        raw_events=db.execute('SELECT payload FROM events WHERE frame>=? AND frame<? ORDER BY frame,tid,kind LIMIT 20001',(event_first,last)).fetchall()
        if len(raw_events)>20000:raise WindowTooDense('Event history exceeds response budget; observations remain in the index')
        events=[json.loads(r[0]) for r in raw_events]
        by_frame={}
        for event in events:
            if event['kind']=='boundary' and event['frame']>=first:by_frame.setdefault(event['frame'],[]).append(event)
        for frame in frames:
            for event in by_frame.get(frame['frame'],[]):
                if event['from_zone']:outgoing[event['from_zone']]+=1
                if event['to_zone']:incoming[event['to_zone']]+=1
            complete=frame['observed'] and (meta['first_uncertain_frame'] is None or frame['frame']<meta['first_uncertain_frame'])
            counts=frame.pop('_counts');frame['flow_complete']=complete
            frame['zones']=[dict(zone=z['name'],count=counts[z['name']] if counts is not None and frame['observed'] else None,
                inflow=incoming[z['name']] if complete else None,outflow=outgoing[z['name']] if complete else None,
                net_inflow=incoming[z['name']]-outgoing[z['name']] if complete else None,
                confirmed_inflow=incoming[z['name']],confirmed_outflow=outgoing[z['name']]) for z in meta['roi_snapshot']['zones']]
        result.update(events=events,diagnostics=[json.loads(r[0]) for r in db.execute('SELECT payload FROM diagnostics WHERE frame>=? AND frame<? ORDER BY frame,tid',(first,last))],
            routes=[dict(from_zone=a,to_zone=b,transitions=n) for a,b,n in db.execute("SELECT src,dst,count(*) FROM events WHERE src<>'' AND dst<>'' GROUP BY src,dst ORDER BY src,dst")],
            observed_od=[dict(from_zone=a,to_zone=b,tracks=n) for a,b,n in db.execute('SELECT * FROM od ORDER BY src,dst')],
            event_provenance='validated_streaming_original_tracking',window_scoped_events=True)
        memberships=db.execute("SELECT DISTINCT e.src,e.dst,e.tid FROM events e JOIN (SELECT DISTINCT tid FROM boxes WHERE frame>=? AND frame<?) t ON t.tid=e.tid WHERE e.src<>'' AND e.dst<>'' LIMIT 20001",(first,last)).fetchall()
        if len(memberships)>20000:raise WindowTooDense('Route membership exceeds response budget')
        grouped={}
        for src,dst,tid in memberships:grouped.setdefault((src,dst),[]).append(tid)
        result['route_track_ids']=[dict(from_zone=a,to_zone=b,track_ids=ids) for (a,b),ids in grouped.items()]
        result['route_prefix_counts']=[dict(from_zone=a,to_zone=b,transitions=n) for a,b,n in db.execute("SELECT src,dst,count(*) FROM events WHERE src<>'' AND dst<>'' AND frame<? GROUP BY src,dst",(event_first,))]
        result['earlier_events']=[json.loads(r[0]) for r in reversed(db.execute('SELECT payload FROM events WHERE frame<? ORDER BY frame DESC,tid DESC,kind DESC LIMIT 30',(event_first,)).fetchall())]
        if len(json.dumps(result))>MAX_RESPONSE_BYTES:raise WindowTooDense('Window exceeds response budget; request a shorter interval')
        return result
