from datetime import datetime, timezone
from typing import Literal
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import AnalysisJob, Video, VisionResult, VisionZoneMetric, VisionBottleneckScore, VenueRevision, ConcentrationRun, OperatorEvent
from app.services import replay_view, detection_view
from app.services.concentration import estimator, evaluate_history
from app.services.concentration_schema import EstimateRequest
from app.services.forecast_contract import ForecastContract
from routers.venues import Configuration,latest

router=APIRouter(prefix='/api/venues',tags=['Central concentration'])


def configuration(db,venue_id,revision):
    row=db.get(VenueRevision,(venue_id,revision))
    if row is None:raise HTTPException(404,'Venue revision not found')
    # Older revisions remain readable; server-generated evidence is not editable.
    raw={k:v for k,v in row.configuration.items() if k not in ('mapping_roi_snapshots','camera_sources')}
    return Configuration.model_validate(raw).model_dump(mode='json')


def contexts_for(db,config):
    from routers.vision_results import bottleneck_diagnostics
    contexts={}
    for camera in config['cameras']:
        job_id=camera['job_id']
        if not job_id or not camera.get('active', True):continue
        if job_id in contexts:continue
        job=db.get(AnalysisJob,UUID(job_id));result=db.get(VisionResult,UUID(job_id))
        if job is None or result is None or job.status!='succeeded':contexts[job_id]={'error':'result_unavailable'};continue
        source=db.get(Video,job.video_id)
        context={'video_id':str(job.video_id),'analysis_mode':job.analysis_mode,
                 'analysis_profile':result.execution_snapshot.get('analysis_profile','full'),
                 'result_fingerprint':result.fingerprint,'source_sha256':result.execution_snapshot.get('video_sha256'),
                 'source_recorded_at':source.recorded_at.isoformat() if source.recorded_at else None}
        try:context['replay']=replay_view.for_job(job,result)
        except (detection_view.DetectionUnavailable,ValueError,KeyError,TypeError,OverflowError,OSError):context['error']='tracking_or_frame_index_unavailable'
        # Reuse the existing saved metrics and unavailable-window policy.
        metrics=db.scalars(select(VisionZoneMetric).where(VisionZoneMetric.analysis_job_id==job.id,VisionZoneMetric.imported_at==result.imported_at)).all()
        hybrid={(r['sec'],r['zone']):r for r in result.execution_snapshot.get('settings',{}).get('hybrid',{}).get('rows',[])}
        context['metrics']=[{'region_key':m.region_key,'video_offset_ms':m.video_offset_ms,'mean_person_count':float(m.mean_person_count) if m.mean_person_count is not None else None,
            'motion':m.motion,'flow_angle_deg':m.flow_angle_deg,'low_confidence':m.low_confidence,'inflow':m.inflow,'outflow':m.outflow,
            'individual_angle_deg':m.mean_angle_deg,'direction_sample_count':m.direction_sample_count,
            'direction_counts':hybrid.get((m.video_offset_ms//1000,m.region_key),{}).get('direction_counts')} for m in metrics]
        context['scores']=[{'region_key':s.region_key,'window_start_ms':s.window_start_ms,'score':s.score,'severity':s.severity} for s in db.scalars(select(VisionBottleneckScore).where(VisionBottleneckScore.analysis_job_id==job.id,VisionBottleneckScore.imported_at==result.imported_at))]
        context['bottleneck_status']=bottleneck_diagnostics(result,db)
        contexts[job_id]=context
    return contexts


@router.get('/forecast-contract/schema')
def forecast_contract_schema():
    return {'status':'awaiting_team_artifact_no_model_loaded','schema':ForecastContract.model_json_schema(),
            'rule':'Administrative population or venue-total output must never be relabeled as center-zone output.'}


@router.post('/{venue_id}/concentration/estimate',status_code=201)
def estimate(venue_id:str,request:EstimateRequest,db:Session=Depends(get_db)):
    current=latest(db,venue_id)
    if current is None:raise HTTPException(404,'Venue not configured')
    if current.revision!=request.revision:raise HTTPException(409,'Map configuration changed; reload before estimating')
    config=select_target(configuration(db,venue_id,request.revision),request.target_zone_id);contexts=contexts_for(db,config)
    result=estimator.estimate(config,contexts,request.model_dump(mode='json'))
    result.update(venue_id=venue_id,revision=request.revision,venue_name=config['name'],registered_address=config['address'] or None,
        center_zone_id=config['center_zone_id'],source_snapshots={k:{a:v for a,v in c.items() if a in ('video_id','result_fingerprint','source_sha256','analysis_mode','source_recorded_at')} for k,c in contexts.items()})
    row=ConcentrationRun(venue_id=venue_id,revision=request.revision,request=request.model_dump(mode='json'),result=result)
    db.add(row)
    try:db.commit();db.refresh(row)
    except SQLAlchemyError:db.rollback();raise HTTPException(503,'Concentration evidence storage unavailable') from None
    return {'run_id':row.id,'created_at':row.created_at,**result}


@router.post('/{venue_id}/concentration/evaluate')
def evaluate(venue_id:str,request:EstimateRequest,db:Session=Depends(get_db)):
    config=select_target(configuration(db,venue_id,request.revision),request.target_zone_id)
    return {'venue_id':venue_id,'revision':request.revision,**evaluate_history(config,contexts_for(db,config))}


def select_target(config, target):
    """Reuse the existing estimator separately for each target; never mix policies."""
    from copy import deepcopy
    config = deepcopy(config)
    target = target or config.get('center_zone_id')
    if target != config.get('center_zone_id'):
        if target not in config.get('target_policies', {}):
            raise HTTPException(422, 'Prediction target is not configured')
        config['concentration_policy'] = config['target_policies'][target]
    config['center_zone_id'] = target
    active = {c['camera_id'] for c in config['cameras'] if c.get('active', True) and c.get('job_id')}
    config['corridors'] = [g for g in config['corridors'] if g['to_zone_id'] == target and g['camera_id'] in active]
    relevant = {g['camera_id'] for g in config['corridors']} | {config['concentration_policy'].get('center_observer_camera_id')}
    config['cameras'] = [c for c in config['cameras'] if c['camera_id'] in active and (not target or c['camera_id'] in relevant or c['zone_id'] == target)]
    return config


@router.get('/{venue_id}/concentration/runs')
def runs(venue_id:str,db:Session=Depends(get_db)):
    rows=db.scalars(select(ConcentrationRun).where(ConcentrationRun.venue_id==venue_id).order_by(ConcentrationRun.created_at.desc()).limit(30)).all()
    return [{'run_id':r.id,'revision':r.revision,'created_at':r.created_at,'mode':r.result['mode'],'state':r.result['state'],'observed_at':r.result['observed_at']} for r in rows]


class OperatorRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    run_id:UUID
    operator:str=Field(min_length=1,max_length=100)
    zone_id:str|None=None
    verification:Literal['unconfirmed','checking','field_verified']='unconfirmed'
    action:Literal['review','staffing_review','route_review','field_check']='review'
    note:str=Field(default='',max_length=4000)


@router.post('/{venue_id}/operator-events',status_code=201)
def record_event(venue_id:str,request:OperatorRequest,db:Session=Depends(get_db)):
    run=db.get(ConcentrationRun,request.run_id)
    if run is None or run.venue_id!=venue_id:raise HTTPException(404,'Concentration run not found')
    config=configuration(db,venue_id,run.revision)
    if request.zone_id and request.zone_id not in {z['id'] for z in config['zones']}:raise HTTPException(422,'Unknown zone')
    event=OperatorEvent(**request.model_dump())
    db.add(event)
    try:db.commit();db.refresh(event)
    except SQLAlchemyError:db.rollback();raise HTTPException(503,'Operator record unavailable') from None
    return {'id':event.id,'created_at':event.created_at,'verification':event.verification,'run_id':event.run_id}


@router.get('/{venue_id}/operator-events')
def operator_events(venue_id:str,db:Session=Depends(get_db)):
    rows=db.scalars(select(OperatorEvent).join(ConcentrationRun,OperatorEvent.run_id==ConcentrationRun.id).where(ConcentrationRun.venue_id==venue_id).order_by(OperatorEvent.created_at.desc()).limit(50)).all()
    return [{'id':r.id,'run_id':r.run_id,'operator':r.operator,'zone_id':r.zone_id,'verification':r.verification,'action':r.action,'note':r.note,'created_at':r.created_at} for r in rows]
