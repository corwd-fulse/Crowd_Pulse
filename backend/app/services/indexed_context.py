"""Lazy disk observations for the existing multi-camera conditional estimator."""
import json,math
from .replay_index import connect,ensure

ISSUES=('id_gap_or_duplicate','suspected_id_switch','missing_track_id','ambiguous_overlap')

def for_job(job,result):
    index=ensure(job,result)
    with connect(index) as db:
        meta=json.loads(db.execute('SELECT payload FROM metadata').fetchone()[0])
    class Frames:
        def __len__(self):return meta['frame_count']
        def parse(self,row):
            if row is None:return None
            f,t,points,counts,observed=row;counts=json.loads(counts)
            return dict(frame=f,t_ms=t,points=json.loads(points),observed=bool(observed),zones=[dict(zone=z['name'],count=counts.get(z['name']) if counts is not None and observed else None) for z in meta['roi_snapshot']['zones']])
        def at(self,frame):
            with connect(index) as db:return self.parse(db.execute('SELECT frame,t_ms,points,counts,observed FROM frames WHERE frame=?',(frame,)).fetchone())
        def __iter__(self):
            with connect(index) as db:
                for row in db.execute('SELECT frame,t_ms,points,counts,observed FROM frames ORDER BY frame'):yield self.parse(row)
    frames=Frames()
    def issues():
        with connect(index) as db:
            yield from db.execute("SELECT coalesce(json_extract(payload,'$.resumed_frame'),frame),tid FROM diagnostics WHERE json_extract(payload,'$.reason') IN (?,?,?,?) ORDER BY 1",ISSUES)
    def quality(offset,seconds):
        if offset<seconds:return False,'insufficient_history'
        start=math.floor((offset-seconds)*meta['fps']+1e-6);end=math.floor(offset*meta['fps']+1e-6)
        with connect(index) as db:
            if not db.execute('SELECT 1 FROM frames WHERE frame=? AND observed=1',(end,)).fetchone():return False,'unobserved_frame'
            if db.execute('SELECT count(*) FROM frames WHERE frame>=? AND frame<=? AND observed=1',(start,end)).fetchone()[0]!=end-start+1:return False,'missing_frame_index'
            if db.execute("SELECT 1 FROM diagnostics WHERE frame>=? AND frame<=? AND json_extract(payload,'$.reason') IN (?,?,?,?) LIMIT 1",(start,end,*ISSUES)).fetchone():return False,'tracking_discontinuity_or_unidentified_observation'
        return True,None
    return {**meta,'frames':frames,'_frame_at':frames.at,'_issues':issues,'_quality':quality}
