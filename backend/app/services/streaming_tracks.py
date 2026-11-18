"""External-sort tracking postprocessing with the existing direction/transition policy.

Only 31 neighboring observations of one ID are retained. SQLite holds all raw
observations and intermediate votes; no training or visualization sampling.
"""
from collections import deque
from contextlib import contextmanager
from itertools import groupby,islice
from pathlib import Path
import csv, math, shutil, sqlite3, tempfile
import numpy as np
import pandas as pd
from .zone_analysis import _zone_of_point, _has_id_switch, MIN_DWELL_FRAMES, MIN_NET_DISPLACEMENT_PX

@contextmanager
def process(source, zones, fps, output):
    source=Path(source)
    if source.stat().st_size>8*1024**3:raise ValueError('Tracking source exceeds 8 GiB processing budget')
    if shutil.disk_usage(Path(output).parent).free < source.stat().st_size*8+256*1024**2:
        raise ValueError('Insufficient external-sort storage reserve')
    with tempfile.TemporaryDirectory(prefix='tracking-sort-',dir=Path(output).parent) as folder:
        db=sqlite3.connect(str(Path(folder)/'tracks.sqlite'))
        try:
            db.execute('PRAGMA cache_size=-8192');db.execute('PRAGMA temp_store=FILE')
            db.executescript('''CREATE TABLE raw(f INTEGER,t INTEGER,id INTEGER,x REAL,y REAL,z TEXT);
                CREATE TABLE flow(sec INTEGER,z TEXT,kind TEXT);
                CREATE TABLE direction(f INTEGER,t INTEGER,id INTEGER,a REAL,s REAL,reliable INTEGER,z TEXT,dt REAL);''')
            batch=[];n=0
            with source.open(encoding='utf-8-sig',newline='') as stream:
                for r in csv.DictReader(stream):
                    x=(float(r['x1'])+float(r['x2']))/2;y=float(r['y2'])
                    batch.append((int(r['frame']),int(r['t_ms']),int(r['track_id']) if r['track_id'] else None,x,y,_zone_of_point((x,y),zones)))
                    n+=1
                    if n>100_000_000:raise ValueError('Tracking sort row budget exceeded')
                    if len(batch)==2000:db.executemany('INSERT INTO raw VALUES(?,?,?,?,?,?)',batch);batch.clear()
                if batch:db.executemany('INSERT INTO raw VALUES(?,?,?,?,?,?)',batch)
            try:db.executescript('CREATE UNIQUE INDEX raw_track ON raw(id,f) WHERE id IS NOT NULL;CREATE INDEX raw_frame ON raw(f);')
            except sqlite3.IntegrityError:raise ValueError('Duplicate track/frame observations; raw source is preserved') from None
            has_id=db.execute('SELECT 1 FROM raw WHERE id IS NOT NULL LIMIT 1').fetchone() is not None
            max_sec=db.execute('SELECT max(t/1000) FROM raw').fetchone()[0] or 0
            if (max_sec+1)*len(zones)>1_000_000:raise ValueError('Time-series metadata exceeds one million zone-second budget')
            counts={(int(sec),z):round(float(n),1) for sec,z,n in db.execute('SELECT sec,z,avg(n) FROM (SELECT t/1000 sec,z,count(*) n FROM raw WHERE z<>\'\' GROUP BY f,t/1000,z) GROUP BY sec,z')}
            with Path(output).open('w',encoding='utf-8',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['frame','t_ms','track_id','angle_deg','speed_px','reliable'])
                directions=[]
                cursor=db.execute('SELECT f,t,id,x,y,z FROM raw WHERE id IS NOT NULL ORDER BY id,f,rowid')
                for tid, track in groupby(cursor,key=lambda r:r[2]):
                    future=deque();history=deque(maxlen=15);iterator=iter(track)
                    for _ in range(16):
                        r=next(iterator,None)
                        if r is not None:future.append(r)
                    confirmed=None;candidate=None;run=0;candidate_point=None;index=0
                    while future:
                        row=future[0];f,t,_,x,y,z=row
                        neighborhood=list(history)+list(future);points=[(r[3],r[4]) for r in neighborhood]
                        switch=_has_id_switch(points,len(history))
                        if index==0:confirmed=z
                        elif z==confirmed:candidate=None;run=0;candidate_point=None
                        else:
                            if z==candidate:run+=1
                            else:candidate=z;run=1;candidate_point=(x,y)
                            if run>=MIN_DWELL_FRAMES and math.dist(candidate_point,(x,y))>=MIN_NET_DISPLACEMENT_PX:
                                if not switch:
                                    # Legacy uses the first timestamp observed for this frame.
                                    sec=db.execute('SELECT t/1000 FROM raw WHERE f=? ORDER BY rowid LIMIT 1',(f,)).fetchone()[0]
                                    if confirmed:db.execute('INSERT INTO flow VALUES(?,?,?)',(sec,confirmed,'outflow'))
                                    if candidate:db.execute('INSERT INTO flow VALUES(?,?,?)',(sec,candidate,'inflow'))
                                confirmed=candidate;candidate=None;run=0;candidate_point=None
                        if history:
                            recent=list(history)[-5:]+[row];lo=recent[0]
                            reliable=not switch and all(b[0]-a[0]==1 and b[1]>a[1] for a,b in zip(recent,recent[1:]))
                            speed=math.dist((x,y),(lo[3],lo[4])) if reliable else None
                            angle=float(np.degrees(np.arctan2(y-lo[4],x-lo[3]))%360) if reliable and speed>0 else None
                            writer.writerow([f,t,tid,angle if angle is not None else '',speed if speed is not None else '',reliable])
                            directions.append((f,t,tid,angle,speed,reliable,z,(t-lo[1])/1000))
                            if len(directions)==2000:db.executemany('INSERT INTO direction VALUES(?,?,?,?,?,?,?,?)',directions);directions.clear()
                        history.append(future.popleft());r=next(iterator,None)
                        if r is not None:future.append(r)
                        index+=1
                if directions:db.executemany('INSERT INTO direction VALUES(?,?,?,?,?,?,?,?)',directions)
            flow={(sec,z,kind):n for sec,z,kind,n in db.execute('SELECT sec,z,kind,count(*) FROM flow GROUP BY sec,z,kind')}
            observations=pd.DataFrame([dict(sec=sec,zone=z.name,count=counts.get((sec,z.name),0.),inflow=flow.get((sec,z.name,'inflow'),0) if has_id else np.nan,outflow=flow.get((sec,z.name,'outflow'),0) if has_id else np.nan) for sec in range(max_sec+1) for z in zones])
            # SQL aggregates retain the legacy per-observation circular mean.
            db.create_function('sin',1,math.sin);db.create_function('cos',1,math.cos)
            aggregates=[]
            for sec,z,s,c,n in db.execute('SELECT cast(f/? as integer),z,avg(sin(a*?/180)),avg(cos(a*?/180)),count(*) FROM direction WHERE reliable=1 AND a IS NOT NULL AND z<>\'\' GROUP BY cast(f/? as integer),z ORDER BY 1,2',(fps,math.pi,math.pi,fps)):
                aggregates.append(dict(sec=sec,zone=z,mean_angle_deg=float(np.degrees(np.arctan2(s,c))%360),n_tracks=n))
            # One group at a time; median speeds preserve one vote per ID/second.
            db.execute('CREATE TABLE votes(sec INTEGER,z TEXT,a REAL,s REAL)')
            for (sec,z,tid),group in groupby(db.execute("SELECT t/1000,z,id,a,s/dt FROM direction WHERE reliable=1 AND a IS NOT NULL AND z<>'' AND s>=1 AND dt>0 ORDER BY t/1000,z,id,f"),key=lambda r:r[:3]):
                values=list(islice(group,100001))
                if len(values)>100000:raise ValueError('One track-second exceeds aggregation budget')
                rad=np.radians([r[3] for r in values]);x,y=np.cos(rad).mean(),np.sin(rad).mean()
                if np.hypot(x,y)>=.3:db.execute('INSERT INTO votes VALUES(?,?,?,?)',(sec,z,float(np.degrees(np.arctan2(y,x))%360),float(np.median([r[4] for r in values]))))
            db.execute('CREATE INDEX votes_sec ON votes(sec,z)')
            class Votes:
                def get(self,key,default):
                    result=db.execute('SELECT a,s FROM votes WHERE sec=? AND z=? LIMIT 20001',key).fetchall()
                    if len(result)>20000:raise ValueError('One zone-second exceeds vote memory budget')
                    return result or default
            yield observations,pd.DataFrame(aggregates,columns=['sec','zone','mean_angle_deg','n_tracks']),Votes()
        finally:db.close()
