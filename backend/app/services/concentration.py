"""Read-only, causal gate observations and explicitly conditional flow extrapolation.

No ML imports, cross-camera ID stitching, geographic direction from pixel angle,
or claimed accident probability. An unobserved center remains unknown.
"""
from bisect import bisect_left,bisect_right
from collections import Counter
from datetime import datetime, timedelta
import math
from typing import Protocol
from app.services.roi_geometry import cross, intersects

VERSION='central-flow-constant-rate-v1'
HORIZONS=(10,30,60)
ISSUES={'id_gap_or_duplicate','suspected_id_switch','missing_track_id','ambiguous_overlap'}


def frame_at(replay, seconds):
    if seconds<0 or seconds>=replay['frame_count']/replay['fps']:
        return None
    wanted=math.floor(seconds*replay['fps']+1e-6)
    if '_frame_at' in replay:
        row=replay['_frame_at'](wanted)
        return row if row and row['observed'] else None
    frames=replay['frames'];low,high=0,len(frames)-1
    while low<=high:
        mid=(low+high)//2;row=frames[mid]
        if row['frame']==wanted:
            return row if row['observed'] else None
        if row['frame']<wanted:low=mid+1
        else:high=mid-1
    return None


def gate_events(replay, gate, *, end_seconds=None):
    """Finite pixel gate: continuous ID, deadband 3px, two opposite-side samples.

    Confirmation timestamp is the second sample, never an interpolated crossing.
    Track IDs are local to this replay. Existing discontinuity diagnostics split it.
    """
    a,b=gate['boundary'];ref=1 if cross(a,b,gate['center_side'])>0 else -1
    length=math.dist(a,b);states={};events=[]
    bad={(d.get('resumed_frame',d['frame']),d.get('track_id')) for d in replay.get('diagnostics',[]) if d['reason'] in ISSUES} if '_issues' not in replay else set()
    issue_stream=iter(replay['_issues']()) if '_issues' in replay else None
    issue=next(issue_stream,None) if issue_stream is not None else None
    for f in replay['frames']:
        if end_seconds is not None and f['t_ms']>end_seconds*1000:break
        if not f['observed']:
            states.clear();continue
        frequencies=Counter(p['track_id'] for p in f['points'] if p['track_id'] is not None)
        if issue_stream is not None:
            bad=set()
            while issue is not None and issue[0]<=f['frame']:
                if issue[0]==f['frame']:bad.add(issue)
                issue=next(issue_stream,None)
        for p in f['points']:
            tid=p['track_id'];point=[p['x'],p['y']]
            if tid is None or frequencies[tid]!=1 or (f['frame'],tid) in bad or (f['frame'],None) in bad:
                states.pop(tid,None);continue
            previous=states.get(tid)
            if previous is None or previous['frame']!=f['frame']-1:
                previous={'stable':0,'candidate':0,'run':0,'confidence':p['confidence'],'last_point':None,'last_nonzero_side':0,'crosses':False}
            previous['frame']=f['frame']
            distance=cross(a,b,point)/length*ref
            raw_side=1 if distance>0 else -1 if distance<0 else 0
            last_point=previous['last_point']
            if raw_side and last_point is not None:
                last_distance=cross(a,b,last_point)/length*ref
                last_side=1 if last_distance>0 else -1 if last_distance<0 else previous['last_nonzero_side']
                if last_side and last_side!=raw_side:
                    # Only actual adjacent observed segments, not a chord spanning
                    # deadband frames that may have gone around the gate endpoint.
                    previous['crosses']=intersects(last_point,point,a,b)
                if raw_side==previous['stable']:previous['crosses']=False
            if raw_side:previous['last_nonzero_side']=raw_side
            side=1 if distance>3 else -1 if distance< -3 else 0
            if side==0:
                previous['run']=0;previous['confidence']=min(previous['confidence'],p['confidence'])
            elif previous['stable']==0 or side==previous['stable']:
                previous.update(stable=side,run=0,confidence=p['confidence'],crosses=False)
            else:
                if previous['candidate']!=side or previous['run']==0:
                    previous.update(candidate=side,run=1,confidence=min(previous['confidence'],p['confidence']))
                else:
                    previous['run']+=1
                    previous['confidence']=min(previous['confidence'],p['confidence'])
                if previous['run']>=2:
                    if previous['crosses']:
                        events.append({'track_id':tid,'frame':f['frame'],'timestamp_ms':f['t_ms'],
                            'direction':'toward_center' if side==1 else 'away_from_center',
                            'confidence':previous['confidence'],'confidence_kind':'uncalibrated_detection_support'})
                        if len(events)>100000:raise ValueError('Gate event memory budget exceeded; use a shorter capture session')
                    previous.update(stable=side,run=0,confidence=p['confidence'],crosses=False)
            previous['last_point']=point
            states[tid]=previous
        # An absent ID must restart on return under the existing adjacency rule.
        states={tid:state for tid,state in states.items() if state['frame']==f['frame']}
    return events


def window_quality(replay, offset, seconds):
    if '_quality' in replay:return replay['_quality'](offset,seconds)
    if offset<seconds:return False,'insufficient_history'
    start=math.floor((offset-seconds)*replay['fps']+1e-6);end=math.floor(offset*replay['fps']+1e-6)
    if frame_at(replay,offset) is None:return False,'unobserved_frame'
    frames=replay['frames']
    left=bisect_left(frames,start,key=lambda f:f['frame']);right=bisect_right(frames,end,key=lambda f:f['frame'])
    indices=[f['frame'] for f in frames[left:right] if f['observed']]
    if len(indices)!=end-start+1:return False,'missing_frame_index'
    if any(d['reason'] in ISSUES and start<=d['frame']<=end for d in replay.get('diagnostics',[])):
        return False,'tracking_discontinuity_or_unidentified_observation'
    return True,None


def events_until(context,gate,offset):
    # Request-local geometry cache. Future events are always filtered out.
    key=(tuple(map(tuple,gate['boundary'])),tuple(gate['center_side']))
    cache=context.setdefault('_gate_events',{})
    if key not in cache:cache[key]=gate_events(context['replay'],gate)
    events=cache[key]
    end=bisect_right(events,offset*1000,key=lambda e:e['timestamp_ms'])
    return events[:end]


def source_offset(config,camera,request):
    if config['integration_mode']=='synchronized':
        if not request.get('at') or not camera.get('clock_verified') or not camera.get('capture_start') or not camera.get('clock_evidence','').strip():
            return None,'unverified_capture_clock'
        at=datetime.fromisoformat(request['at']);start=datetime.fromisoformat(camera['capture_start'])
        return (at-start).total_seconds(),None
    return request['offset_seconds']+(camera.get('scenario_offset_s',0) if config['integration_mode']=='scenario' else 0),None


def sample_camera(config,camera,context,request):
    offset,reason=source_offset(config,camera,request)
    row={'camera_id':camera['camera_id'],'name':camera['name'],'zone_id':camera['zone_id'],
         'job_id':camera['job_id'],'video_id':context.get('video_id'),'offset_seconds':offset,
         'count':None,'count_change':None,'metric':None,'bottleneck':None,'reason':reason or context.get('error'),
         'analysis_mode':context.get('analysis_mode'),'observed_at':None}
    replay=context.get('replay')
    if replay is None or offset is None:return row
    f=frame_at(replay,offset)
    zone=next((z for z in f['zones'] if z['zone']==camera['roi_name']),None) if f else None
    row['count']=zone['count'] if zone else None
    if row['count'] is None:row['reason']='unobserved_frame_or_roi'
    if camera.get('capture_start') and camera.get('clock_verified'):
        row['observed_at']=(datetime.fromisoformat(camera['capture_start'])+timedelta(seconds=offset)).isoformat()
    old=frame_at(replay,offset-config['concentration_policy']['window_seconds'])
    oldzone=next((z for z in old['zones'] if z['zone']==camera['roi_name']),None) if old else None
    if oldzone and oldzone['count'] is not None and row['count'] is not None:row['count_change']=row['count']-oldzone['count']
    # Only completed seconds/windows; no future bucket enters the estimate.
    row['metric']=next((m for m in context.get('metrics',[]) if m['region_key']==camera['roi_name'] and m['video_offset_ms']//1000==math.floor(offset)-1),None)
    scores=[s for s in context.get('scores',[]) if s['region_key']==camera['roi_name'] and s['window_start_ms']/1000+10<=offset]
    if scores:
        score=max(scores,key=lambda s:s['window_start_ms'])
        status=next((s for s in context.get('bottleneck_status',[]) if s['zone']==camera['roi_name'] and s['window_start']*1000==score['window_start_ms']),None)
        missing=status and (status['status']=='unanalyzed' or any(status.get('missing_metrics',{}).values()) or any(status.get('previous_missing_metrics',{}).values()) or status.get('flow_low_confidence_seconds'))
        if not missing and context.get('analysis_profile')!='fast':row['bottleneck']=score
    return row


class Estimator(Protocol):
    version: str
    def estimate(self, configuration:dict, contexts:dict, request:dict)->dict: ...


class ConstantFlowEstimator:
    version=VERSION

    def estimate(self,config,contexts,request):
        policy=config['concentration_policy'];window=policy['window_seconds'];mode=config['integration_mode']
        cameras={c['camera_id']:c for c in config['cameras']}
        samples=[sample_camera(config,c,contexts.get(c['job_id'],{'error':'result_unavailable'}),request) for c in cameras.values()]
        sample_by_id={r['camera_id']:r for r in samples};routes=[]
        for gate in config['corridors']:
            camera=cameras[gate['camera_id']];sample=sample_by_id[gate['camera_id']]
            context=contexts.get(camera['job_id'],{});replay=context.get('replay');offset=sample['offset_seconds']
            events=events_until(context,gate,offset) if gate['boundary_verified'] and replay and offset is not None and offset>=0 else []
            recent=[e for e in events if (offset-window)*1000<e['timestamp_ms']<=offset*1000] if offset is not None else []
            before=[e for e in events if (offset-2*window)*1000<e['timestamp_ms']<=(offset-window)*1000] if offset is not None else []
            quality,reason=window_quality(replay,offset,window) if replay and offset is not None else (False,sample['reason'] or 'replay_unavailable')
            if not gate['boundary_verified']:quality,reason=False,'boundary_not_verified'
            toward=sum(e['direction']=='toward_center' for e in recent);away=len(recent)-toward
            previous=sum(e['direction']=='toward_center' for e in before)
            trend=toward-previous if quality and offset>=2*window and window_quality(replay,offset-window,window)[0] else None
            routes.append({'corridor_id':gate['corridor_id'],'name':gate['name'],'camera_id':gate['camera_id'],
                'from_zone_id':gate['from_zone_id'],'to_zone_id':gate['to_zone_id'],'counting_group':gate['counting_group'],
                'toward_center':toward if sample['count'] is not None and gate['boundary_verified'] else None,'away_from_center':away if sample['count'] is not None and gate['boundary_verified'] else None,
                'net':toward-away if sample['count'] is not None and gate['boundary_verified'] else None,'toward_change':trend,'window_seconds':window,
                'offset_seconds':offset,'observed_at':sample['observed_at'],'quality':'usable_low_confidence' if quality else 'unavailable',
                'reason':reason,'role':gate['role'],'arrival_basis':'direct_boundary' if gate['role']=='center_boundary' else gate['arrival_basis'],
                'arrival_ratio':1 if gate['role']=='center_boundary' else gate['arrival_ratio'] if gate['arrival_basis']!='unknown' else None,
                'travel_seconds':0 if gate['role']=='center_boundary' else gate['travel_seconds'] if gate['arrival_basis']!='unknown' else None,
                'source':'observed_finite_gate_crossings','confirmed_events':recent,
                'events_are_lower_bounds':True,'event_namespace':f"{camera['camera_id']}:{camera['job_id']}"})
        reasons=[]
        if mode=='independent':reasons.append('independent_videos_not_aggregated')
        if not config.get('center_zone_id'):reasons.append('center_not_configured')
        if not routes:reasons.append('no_center_corridors')
        gate_camera_ids={r['camera_id'] for r in routes}
        if any(s['reason'] for s in samples if s['camera_id'] in gate_camera_ids):reasons.append('camera_unobserved_or_unsynchronized')
        if mode=='synchronized' and any(s['analysis_mode']!='production' for s in samples):reasons.append('test_analysis_not_real_simultaneous_observation')
        # Reusing the same video/ROI through two camera IDs can never add people.
        sources=[(contexts.get(c['job_id'],{}).get('source_sha256') or contexts.get(c['job_id'],{}).get('video_id',c['job_id']),c['roi_name']) for c in cameras.values()]
        if len(set(sources))!=len(sources):reasons.append('duplicate_video_roi_source')
        groups=[r['counting_group'] for r in routes]
        if len(set(groups))!=len(groups):reasons.append('overlapping_counting_groups')
        evidence_keys=[]
        for r in routes:
            camera=cameras[r['camera_id']];context=contexts.get(camera['job_id'],{})
            source=context.get('source_sha256') or context.get('video_id',camera['job_id'])
            evidence_keys.extend((source,e['track_id'],e['frame'],e['direction']) for e in r['confirmed_events'])
        if len(set(evidence_keys))!=len(evidence_keys):reasons.append('duplicate_crossing_evidence')
        if any(not g['independent_counting_verified'] or not g['independence_evidence'].strip() for g in config['corridors']):reasons.append('counting_independence_unverified')
        if any(r['quality']=='unavailable' for r in routes):reasons.append('incomplete_gate_history')
        common_at=request.get('at') if mode=='synchronized' and not any(r in reasons for r in ('camera_unobserved_or_unsynchronized','test_analysis_not_real_simultaneous_observation')) else None
        observer=cameras.get(policy.get('center_observer_camera_id'));central=sample_by_id.get(observer['camera_id']) if observer else None
        center_count=central['count'] if central else None
        can_total=bool(center_count is not None and observer.get('full_zone_coverage') and policy['center_boundary_coverage_complete'] and policy['coverage_evidence'].strip() and routes and all(r['role']=='center_boundary' for r in routes))
        estimates=[]
        for horizon in HORIZONS:
            inflow=outflow=None
            arrival_ready=all(r['arrival_ratio'] is not None and r['travel_seconds'] is not None for r in routes)
            arrival_history=arrival_ready and all(window_quality(contexts[cameras[r['camera_id']]['job_id']]['replay'],r['offset_seconds'],window+r['travel_seconds'])[0] for r in routes if r['quality']!='unavailable')
            if not reasons and arrival_ready and arrival_history:
                inflow=0.;outflow=0. if all(r['role']=='center_boundary' for r in routes) else None
                for r in routes:
                    ratio=r['arrival_ratio'];delay=r['travel_seconds'];offset=r['offset_seconds']
                    gate=next(g for g in config['corridors'] if g['corridor_id']==r['corridor_id']);camera=cameras[r['camera_id']]
                    # Known approach crossings still in transit plus conditional future crossings.
                    past=events_until(contexts[camera['job_id']],gate,offset)
                    inflight=sum(e['direction']=='toward_center' and offset<e['timestamp_ms']/1000+delay<=offset+horizon for e in past)
                    inflow+=ratio*(inflight+r['toward_center']/window*max(0,horizon-delay))
                    if outflow is not None:outflow+=r['away_from_center']/window*horizon
            total=max(0,center_count+inflow-outflow) if can_total and inflow is not None and outflow is not None else None
            estimates.append({'horizon_seconds':horizon,'target_at':(datetime.fromisoformat(common_at)+timedelta(seconds=horizon)).isoformat() if common_at else None,
                'target_offset_seconds':request['offset_seconds']+horizon,'expected_arrivals':round(inflow,2) if inflow is not None else None,
                'expected_departures':round(outflow,2) if outflow is not None else None,'expected_center_count':round(total,2) if total is not None else None,
                'inward_pressure_per_minute':round(sum(r['toward_center']/window*60 for r in routes),2) if not reasons else None,
                'confidence':'low_conditional_unvalidated' if not reasons else 'unavailable',
                'reasons':reasons+([] if arrival_ready else ['arrival_ratio_or_travel_time_unknown'])+(['insufficient_arrival_history'] if arrival_ready and not arrival_history else [])+([] if can_total else ['center_total_unavailable']),
                'validation_status':'not_validated_on_real_simultaneous_cameras'})
        state='미분석';basis=[]
        if not reasons:
            state='정상' if can_total and policy['thresholds_validated'] and policy['attention_net_per_minute'] is not None else '미분석'
            net=sum((r['net'] or 0)/window*60 for r in routes)
            if policy['attention_net_per_minute'] is not None and net>=policy['attention_net_per_minute']:
                state='주의';basis.append('configured_net_inflow_threshold')
            if sum(r['toward_change'] is not None and r['toward_change']>0 for r in routes)>=2:
                state='주의';basis.append('multiple_corridors_increasing')
            if any(s.get('bottleneck',{}).get('severity')=='high' for s in samples if s.get('bottleneck')):
                state='경계';basis.append('stored_bottleneck_high_research_indicator')
            center_zone=next((z for z in config['zones'] if z['id']==config.get('center_zone_id')),None)
            if center_zone and center_zone.get('capacity') and can_total:
                ratio=center_count/center_zone['capacity']
                if policy['caution_capacity_ratio'] and ratio>=policy['caution_capacity_ratio']:state='경계';basis.append('configured_observed_capacity_ratio')
                if policy['urgent_capacity_ratio'] and ratio>=policy['urgent_capacity_ratio']:state='긴급 확인 필요';basis.append('configured_urgent_observed_ratio')
        priority=sorted(routes,key=lambda r:-(r['toward_center'] if r['toward_center'] is not None else -1))
        return {'model_version':self.version,'method':'conditional_constant_flow_not_trained_forecast',
            'gate_policy':{'deadband_pixels':3,'opposite_side_confirmation_samples':2,'finite_segment':True,'cross_camera_ids_linked':False},
            'mode':mode,'observation_kind':'virtual_scenario' if mode=='scenario' else 'recorded_video_observation',
            'observed_at':common_at,'requested_alignment_at':request.get('at'),'reference_offset_seconds':request['offset_seconds'],'window_seconds':window,
            'cameras':samples,'routes':routes,'observed_center_count':center_count,'center_total_coverage_verified':can_total,
            'aggregation_allowed':not reasons,'aggregation_reasons':reasons,'estimates':estimates,
            'state':state,'state_basis':basis,'threshold_basis':'administrator_validated' if policy['thresholds_validated'] else 'administrator_proposal_not_safety_standard',
            'increasing_corridors':sum(r['toward_change'] is not None and r['toward_change']>0 for r in routes) if not reasons else None,
            'priority_corridors':[{'corridor_id':r['corridor_id'],'name':r['name'],'toward_center':r['toward_center'],
                'recommendation':'영상·현장 확인 후 사전 배치·동선 조정 검토' if r['toward_center'] is not None else '관측·동기화 상태 추가 확인'} for r in priority],
            'limitations':['Confirmed crossings are observation lower bounds, not unique visitors.',
                'Constant flow is an unvalidated assumption; no accident probability or staffing count.',
                'Approach direction is not confirmed center arrival; no cross-camera track stitching.']}


estimator:Estimator=ConstantFlowEstimator()


def evaluate_history(config,contexts,*,max_samples=50):
    """Retrospective ROI-count comparison, not event-level real-world ground truth."""
    policy=config['concentration_policy'];observer=next((c for c in config['cameras'] if c['camera_id']==policy.get('center_observer_camera_id')),None)
    empty=[{'horizon_seconds':h,'samples':0,'mae':None,'rmse':None,'status':'insufficient_observed_center_history'} for h in HORIZONS]
    if not observer or not config['cameras']:return {'model_version':VERSION,'metrics':empty,'evidence':'no_ground_truth_supplied'}
    rows=[]
    for horizon in HORIZONS:
        errors=[];anchors=[]
        durations=[contexts.get(c['job_id'],{}).get('replay') for c in config['cameras']]
        if any(r is None for r in durations):rows.append(next(r for r in empty if r['horizon_seconds']==horizon));continue
        # synchronized recordings may have different first-frame clocks.
        if config['integration_mode']=='synchronized':
            if any(not c.get('capture_start') for c in config['cameras']):rows.append(next(r for r in empty if r['horizon_seconds']==horizon));continue
            starts=[datetime.fromisoformat(c['capture_start']) for c in config['cameras']]
            first=max(starts)+timedelta(seconds=policy['window_seconds'])
            last=min(s+timedelta(seconds=r['frame_count']/r['fps']) for s,r in zip(starts,durations))-timedelta(seconds=horizon+1)
            available=int((last-first).total_seconds())
        else:
            available=int(min(r['frame_count']/r['fps']-c.get('scenario_offset_s',0) for c,r in zip(config['cameras'],durations)))-horizon-policy['window_seconds']-1
        step=max(1,math.ceil(max(0,available)/max_samples))
        for delta in range(0,max(0,available)+1,step):
            offset=policy['window_seconds']+delta
            req={'offset_seconds':offset,'at':(first+timedelta(seconds=delta)).isoformat() if config['integration_mode']=='synchronized' else None}
            result=estimator.estimate(config,contexts,req);prediction=next(e for e in result['estimates'] if e['horizon_seconds']==horizon)['expected_center_count']
            if prediction is None:continue
            target=dict(req);target['offset_seconds']+=horizon
            if req['at']:target['at']=(datetime.fromisoformat(req['at'])+timedelta(seconds=horizon)).isoformat()
            actual=sample_camera(config,observer,contexts[observer['job_id']],target)['count']
            if actual is not None:errors.append(prediction-actual);anchors.append(req['at'] or offset)
        rows.append({'horizon_seconds':horizon,'samples':len(errors),'mae':round(sum(abs(e) for e in errors)/len(errors),4) if errors else None,
            'rmse':round(math.sqrt(sum(e*e for e in errors)/len(errors)),4) if errors else None,
            'status':'retrospective_roi_comparison' if errors else 'insufficient_observed_center_history','anchors':anchors})
    return {'model_version':VERSION,'metrics':rows,'mode':config['integration_mode'],
            'evidence':'observed_roi_counts_not_independently_labeled_ground_truth','max_samples_per_horizon':max_samples,
            'five_minute_supported':False}
