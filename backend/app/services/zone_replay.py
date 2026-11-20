"""Observed track replay and conservative transitions; no inference/interpolation."""
from collections import defaultdict,Counter
import math
from .roi_geometry import inside,signed_distance

VERSION='observed-zone-transitions-v1'
def build_replay(detections,roi,*,observed_frames=None,min_dwell=3,hysteresis_px=3.,min_displacement_px=15.):
    zones=roi['zones'];fps=detections['fps'];frame_count=detections['frame_count']
    tracks=defaultdict(list);frames=[];diagnostics=[];events=[];segments=defaultdict(list);fragmented=set()
    known=set(observed_frames) if observed_frames is not None else set()
    def classify(p):
        matches=[z['name'] for z in zones if inside(p,z['polygon'])]
        return matches[0] if len(matches)==1 else '' if not matches else None
    for f in detections['frames']:
        points=[];counts={z['name']:0 for z in zones};ambiguous=False
        for box in f['boxes']:
            p=[(box['x1']+box['x2'])/2,box['y2']];zone=classify(p)
            if zone in counts:counts[zone]+=1
            if zone is None:ambiguous=True
            point=dict(track_id=box['track_id'],x=p[0],y=p[1],zone=zone,confidence=box['confidence'])
            points.append(point)
            if box['track_id'] is not None:tracks[box['track_id']].append((f['frame'],f['t_ms'],p,zone,box['confidence']))
            else:diagnostics.append(dict(frame=f['frame'],track_id=None,reason='missing_track_id'))
        frames.append(dict(frame=f['frame'],t_ms=f['t_ms'],points=points,
                           observed=f['frame'] in known,counts=None if f['frame'] not in known or ambiguous else counts))
    present={f['frame'] for f in frames}
    for f in sorted(known-present):
        frames.append(dict(frame=f,t_ms=int(f/fps*1000),points=[],observed=True,counts={z['name']:0 for z in zones}))
    polygons={z['name']:z['polygon'] for z in zones}
    from .replay_state import TrackState
    od=Counter()
    for tid,observations in tracks.items():
        state=TrackState(tid,polygons,events.append,diagnostics.append,min_dwell=min_dwell,hysteresis=hysteresis_px,displacement=min_displacement_px)
        for observation in observations:state.step(*observation)
        pair=state.finish(frame_count)
        if pair:od[pair]+=1
        if state.fragmented:fragmented.add(tid)
    events.sort(key=lambda e:(e['frame'],e['track_id'],e['kind']))
    routes=Counter((e['from_zone'],e['to_zone']) for e in events if e['from_zone'] and e['to_zone'])
    boundaries=[e for e in events if e['kind']=='boundary']
    by_frame=defaultdict(list)
    for e in boundaries:by_frame[e['frame']].append(e)
    inflow=Counter();outflow=Counter();uncertainty=sorted(d['frame'] for d in diagnostics);uncertain_at=uncertainty[0] if uncertainty else frame_count
    for f in sorted(frames,key=lambda r:r['frame']):
        for e in by_frame[f['frame']]:
            if e['from_zone']:outflow[e['from_zone']]+=1
            if e['to_zone']:inflow[e['to_zone']]+=1
        complete=f['observed'] and f['frame']<uncertain_at
        f['flow_complete']=complete
        f['zones']=[dict(zone=z['name'],count=None if f['counts'] is None else f['counts'][z['name']],
            inflow=inflow[z['name']] if complete else None,outflow=outflow[z['name']] if complete else None,
            net_inflow=inflow[z['name']]-outflow[z['name']] if complete else None,
            confirmed_inflow=inflow[z['name']],confirmed_outflow=outflow[z['name']]) for z in zones]
        del f['counts']
    diagnostics.sort(key=lambda d:(d['frame'],str(d['track_id'])))
    return dict(version=VERSION,analysis_job_id=detections['analysis_job_id'],video_id=detections['video_id'],
        fps=fps,frame_count=frame_count,video_resolution=detections['video_resolution'],roi_snapshot=roi,
        mode='observed_replay_not_prediction',policy=dict(min_dwell_frames=min_dwell,hysteresis_px=hysteresis_px,
            min_displacement_px=min_displacement_px,overlap='ambiguous_unanalyzed',gap='observed_outside',
            id_discontinuity='split_without_linking',visibility='display_only'),
        frames=sorted(frames,key=lambda r:r['frame']),events=events,diagnostics=diagnostics,
        routes=[dict(from_zone=a,to_zone=b,transitions=n) for (a,b),n in sorted(routes.items())],
        observed_od=[dict(from_zone=a,to_zone=b,tracks=n) for (a,b),n in sorted(od.items())],
        od_note='One start/end pair per unfragmented observed ID with zone change; censored observations, not true journey OD',
        coverage=dict(verified_frame_count=len(known),complete_frame_index=len(known)==frame_count,
                      incomplete_track_ids=len(fragmented),unresolved_paths=len(diagnostics)))
