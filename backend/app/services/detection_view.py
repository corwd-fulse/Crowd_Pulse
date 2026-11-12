"""Read-only view of existing worker tracking output. Never imports the AI."""
import csv
import json
import math
from functools import lru_cache

from app.services import vision_results as storage


class DetectionUnavailable(Exception):
    pass


def for_job(job, result):
    if job.status != 'succeeded' or result is None or job.worker_token is None:
        raise DetectionUnavailable('Job has no completed tracking run')
    root = (storage.BASE_DIR / 'uploads/analysis' / str(job.id)).resolve()
    allowed = (storage.BASE_DIR / 'uploads/analysis').resolve()
    if not root.is_relative_to(allowed):
        raise DetectionUnavailable('Tracking directory unavailable')
    matches = []
    candidates = list(root.glob(f'{job.worker_token.hex}-*/tracker/video_detections.csv'))
    if not candidates:
        raise DetectionUnavailable('Tracking CSV not found for this worker run')
    for path in candidates:
        path = path.resolve()
        if not path.is_relative_to(root):
            continue
        meta = (path.parent.parent / 'execution_metadata.json').resolve()
        if not meta.is_relative_to(root):
            continue
        try:
            if json.loads(meta.read_text(encoding='utf-8')) == result.execution_snapshot:
                matches.append(path)
        except (OSError, ValueError):
            raise ValueError('Tracking execution metadata could not be read') from None
    if len(matches) != 1:
        raise DetectionUnavailable('Matching tracking file or execution metadata unavailable')
    path = matches[0]
    fps = float(result.execution_snapshot['fps'])
    count = int(result.execution_snapshot['frame_count'])
    if not math.isfinite(fps) or fps <= 0 or count <= 0 or count > 500_000:
        raise ValueError('Invalid video metadata')
    stat = path.stat()
    if stat.st_size > 64 * 1024 * 1024:
        raise ValueError('Tracking CSV exceeds view limit')
    frames = read_frames(str(path), stat.st_mtime_ns, stat.st_size, fps, count, result.width, result.height)
    return {'analysis_job_id': str(job.id), 'video_id': str(job.video_id),
            'fps': fps, 'frame_count': count, 'video_resolution': [result.width, result.height],
            'time_unit': 'video-relative milliseconds', 'frames': frames}


@lru_cache(maxsize=4)
def read_frames(path, mtime, size, fps, count, width, height):
    frames = {}
    with open(path, newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ['frame','t_ms','track_id','x1','y1','x2','y2','conf']:
            raise ValueError('Invalid detection columns')
        for i, row in enumerate(reader):
            if i >= 500_000:
                raise ValueError('Tracking row limit exceeded')
            frame, t_ms = int(row['frame']), int(row['t_ms'])
            track = int(row['track_id']) if row['track_id'] else None
            x1,y1,x2,y2,conf = (float(row[k]) for k in ['x1','y1','x2','y2','conf'])
            if not (0 <= frame < count and abs(t_ms - int(frame/fps*1000)) <= 1
                    and (track is None or track >= 0) and all(math.isfinite(v) for v in [x1,y1,x2,y2,conf])
                    and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and 0 <= conf <= 1):
                raise ValueError('Invalid detection row')
            bucket = frames.setdefault(frame, {'frame': frame, 't_ms': t_ms, 'boxes': []})
            if bucket['t_ms'] != t_ms:
                raise ValueError('Inconsistent detection timestamp')
            bucket['boxes'].append({'track_id': track, 'x1':x1,'y1':y1,'x2':x2,'y2':y2,'confidence':conf})
    return [frames[k] for k in sorted(frames)]
