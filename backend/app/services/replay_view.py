"""Immutable worker CSV adapter for the read-only observed replay endpoint."""
import csv,json
from functools import lru_cache
from pathlib import Path
from . import detection_view
from .zone_replay import build_replay

def for_job(job,result):
    path=detection_view.tracking_path(job,result)
    if path.stat().st_size>8*1024**2 or result.execution_snapshot['frame_count']>500000:
        from .indexed_context import for_job as indexed
        return indexed(job,result)
    frames=path.with_name('video_frames.csv')
    a=path.stat();b=frames.stat() if frames.is_file() else None
    result_data=cached(str(path),a.st_mtime_ns,a.st_size,str(frames),b.st_mtime_ns if b else None,b.st_size if b else None,
                  float(result.execution_snapshot['fps']),int(result.execution_snapshot['frame_count']),
                  result.width,result.height,json.dumps(result.roi_snapshot,sort_keys=True),str(job.id),str(job.video_id))
    recorded=result.execution_snapshot.get('settings',{}).get('transitions')
    if recorded and (recorded.get('version')!=result_data['version'] or recorded.get('events')!=result_data['events']):
        raise ValueError('Recorded transition policy differs from replay; do not silently reinterpret')
    return {**result_data,'event_provenance':'recorded_in_execution_snapshot' if recorded and recorded.get('version')==result_data['version'] else 'derived_read_only_from_immutable_tracking'}

@lru_cache(maxsize=4)
def cached(path,mtime,size,frame_path,frame_mtime,frame_size,fps,count,width,height,roi_json,job_id,video_id):
    if size>64*1024*1024 or count>500_000:raise ValueError('Replay input limit exceeded')
    data=detection_view.read_frames(path,mtime,size,fps,count,width,height)
    known=None
    if frame_size is not None:
        if frame_size>32*1024*1024:raise ValueError('Frame index too large')
        actual={f['frame']:len(f['boxes']) for f in data};known=set()
        with Path(frame_path).open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream)
            if reader.fieldnames!=['frame','t_ms','count']:raise ValueError('Invalid frame index')
            for row in reader:
                frame,t_ms,n=(int(row[k]) for k in ['frame','t_ms','count'])
                if (frame in known or not 0<=frame<count or abs(t_ms-int(frame/fps*1000))>1 or
                        n!=actual.get(frame,0)):raise ValueError('Frame/detection mismatch')
                known.add(frame)
        if len(known)!=count:raise ValueError('Incomplete frame index')
    detection=dict(analysis_job_id=job_id,video_id=video_id,fps=fps,frame_count=count,
                   video_resolution=[width,height],frames=data)
    return build_replay(detection,json.loads(roi_json),observed_frames=known)
