"""Durable event summary for NEW pipeline runs; five CSV schemas stay unchanged."""
from pathlib import Path
import csv
from collections import defaultdict
from .zone_replay import build_replay

def summarize_tracking(detections_csv,zones,fps,frame_count,width,height):
    if Path(detections_csv).stat().st_size>8*1024**2:
        import json,tempfile
        from .replay_index import build,connect
        source=Path(detections_csv);frame_path=source.with_name('video_frames.csv')
        stat=source.stat();fs=frame_path.stat() if frame_path.is_file() else None
        meta=dict(version='observed-zone-transitions-v1',analysis_job_id='pipeline',video_id='source',fps=fps,
                  frame_count=frame_count,video_resolution=[width,height],roi_snapshot=dict(zones=[dict(name=z.name,polygon=z.polygon) for z in zones]))
        with tempfile.TemporaryDirectory(prefix='transitions-',dir=source.parent) as folder:
            destination=Path(folder)/'index.sqlite'
            build(source,frame_path,meta,destination,(stat.st_size,stat.st_mtime_ns),(fs.st_size,fs.st_mtime_ns) if fs else None)
            with connect(destination) as db:
                result=json.loads(db.execute('SELECT payload FROM metadata').fetchone()[0])
                for table in ('events','diagnostics'):
                    size=db.execute(f'SELECT sum(length(payload)) FROM {table}').fetchone()[0] or 0
                    if size>12*1024**2:raise ValueError('Transition metadata exceeds bounded summary budget; raw tracking is preserved')
                    order='frame,tid,kind' if table=='events' else 'frame,cast(tid as text)'
                    result[table]=[json.loads(r[0]) for r in db.execute(f'SELECT payload FROM {table} ORDER BY {order}')]
                result['routes']=[dict(from_zone=a,to_zone=b,transitions=n) for a,b,n in db.execute("SELECT src,dst,count(*) FROM events WHERE src<>'' AND dst<>'' GROUP BY src,dst ORDER BY src,dst")]
                result['observed_od']=[dict(from_zone=a,to_zone=b,tracks=n) for a,b,n in db.execute('SELECT * FROM od ORDER BY src,dst')]
                return {k:result[k] for k in ['version','policy','events','routes','observed_od','od_note','diagnostics','coverage']}
    frames=defaultdict(list);times={}
    with Path(detections_csv).open(encoding='utf-8-sig',newline='') as stream:
        for r in csv.DictReader(stream):
            times[int(r['frame'])]=int(r['t_ms'])
            frames[int(r['frame'])].append(dict(track_id=int(r['track_id']) if r['track_id'] else None,
                x1=float(r['x1']),y1=float(r['y1']),x2=float(r['x2']),y2=float(r['y2']),confidence=float(r['conf'])))
    index=Path(detections_csv).with_name('video_frames.csv');known=None
    if index.is_file():
        known=set()
        with index.open(encoding='utf-8-sig',newline='') as stream:
            for r in csv.DictReader(stream):
                f=int(r['frame'])
                if f in known or not 0<=f<frame_count or int(r['count'])!=len(frames.get(f,[])):raise ValueError('Invalid tracker frame index')
                known.add(f)
        if len(known)!=frame_count:raise ValueError('Incomplete tracker frame index')
    data=dict(analysis_job_id='pipeline',video_id='source',fps=fps,frame_count=frame_count,video_resolution=[width,height],
        frames=[dict(frame=f,t_ms=times[f],boxes=boxes) for f,boxes in sorted(frames.items())])
    result=build_replay(data,dict(video_resolution=[width,height],zones=[dict(name=z.name,polygon=z.polygon) for z in zones]),observed_frames=known)
    return {k:result[k] for k in ['version','policy','events','routes','observed_od','od_note','diagnostics','coverage']}
