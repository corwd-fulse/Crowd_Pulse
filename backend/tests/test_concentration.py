"""Controlled synthetic traces only; not real multi-camera performance claims."""
from copy import deepcopy
from datetime import datetime,timezone
from uuid import UUID,uuid4
import pytest
from pydantic import ValidationError
from sqlalchemy import select
from app.services.concentration import gate_events,estimator,evaluate_history,frame_at
from app.services.concentration_schema import Corridor
from app.services.forecast_contract import ForecastContract,validate_center_output
from routers.venues import Configuration
from app.models import ConcentrationRun,OperatorEvent
from test_vision_results import client,seeded,bundle,upload


def trace(xs=None,seconds=160):
    xs=xs or [20+min(i%80,80-i%80)*2 for i in range(seconds)]
    return {'analysis_job_id':'job','video_id':'video','fps':1,'frame_count':len(xs),'video_resolution':[120,120],
        'frames':[{'frame':i,'t_ms':i*1000,'observed':True,'points':[{'track_id':1,'x':x,'y':50,'zone':'C','confidence':.9}],
                   'zones':[{'zone':'C','count':10}]} for i,x in enumerate(xs)],'diagnostics':[]}


def config_and_contexts(count=1):
    zones=[{'id':'a','name':'A','polygon':[[0,0],[100,0],[100,100]],'color':'#44b6c8'},
           {'id':'center','name':'CENTER','polygon':[[400,400],[600,400],[600,600]],'color':'#44b6c8'}]
    cameras=[];gates=[];contexts={}
    for i in range(count):
        job=str(UUID(int=i+1));cam=f'cam{i+1}'
        cameras.append(dict(camera_id=cam,name=cam,point=[i*50,0],zone_id='center',job_id=job,roi_name='C',full_zone_coverage=True,
            capture_start=f'2026-11-17T00:00:0{i}+09:00',clock_verified=True,clock_evidence='synthetic clock fixture'))
        gates.append(dict(corridor_id=f'g{i}',name=f'Gate {i}',camera_id=cam,from_zone_id='a',to_zone_id='center',
            boundary=[[50,0],[50,100]],center_side=[80,50],role='center_boundary',boundary_verified=True,boundary_evidence='synthetic physical boundary fixture',
            counting_group=f'unique{i}',independent_counting_verified=True,independence_evidence='synthetic distinct entrances'))
        contexts[job]={'replay':trace(),'video_id':f'video{i}','analysis_mode':'production','metrics':[],'scores':[]}
    config=Configuration(name='SYNTHETIC ONLY',zones=zones,center_zone_id='center',integration_mode='scenario',cameras=cameras,corridors=gates,
        concentration_policy={'window_seconds':10,'center_observer_camera_id':'cam1','center_boundary_coverage_complete':True,'coverage_evidence':'synthetic full boundary coverage'}).model_dump(mode='json')
    return config,contexts


def request(offset=25,at=None):return dict(offset_seconds=offset,at=at)


def test_finite_gate_both_directions_and_deadband():
    config,_=config_and_contexts();gate=config['corridors'][0]
    replay=trace([20,25,30,40,46,49,51,54,58,65,60,54,51,49,46,42])
    events=gate_events(replay,gate)
    assert [e['direction'] for e in events]==['toward_center','away_from_center']
    assert [e['frame'] for e in events]==[8,15]
    outside=deepcopy(replay)
    for f in outside['frames']:f['points'][0]['y']=110
    assert not gate_events(outside,gate)
    assert not gate_events(trace([49,51]*20),gate)
    around=trace([40,46,49,51,54,58])
    around['frames'][2]['points'][0]['y']=10
    around['frames'][3]['points'][0]['y']=10
    short_gate={**gate,'boundary':[[50,20],[50,100]]}
    assert not gate_events(around,short_gate),'Deadband trajectory around the endpoint must not become a straight chord crossing'


def test_gap_duplicate_id_and_suspected_switch_do_not_link():
    config,_=config_and_contexts();gate=config['corridors'][0]
    base=trace([20,30,40,60,70,80])
    for kind in ['gap','duplicate','switch']:
        r=deepcopy(base)
        if kind=='gap':r['frames'][2]['points']=[]
        if kind=='duplicate':r['frames'][3]['points']*=2
        if kind=='switch':r['diagnostics']=[dict(frame=3,track_id=1,reason='suspected_id_switch')]
        assert not gate_events(r,gate),kind


def test_actual_boundary_count_does_not_follow_optical_angle():
    config,ctx=config_and_contexts();source=next(iter(ctx.values()))
    source['metrics']=[dict(region_key='C',video_offset_ms=24000,flow_angle_deg=0,motion=1,low_confidence=True)]
    result=estimator.estimate(config,ctx,request())
    assert result['routes'][0]['toward_center']==1
    assert result['estimates'][0]['expected_center_count']==11
    assert result['cameras'][0]['metric']['low_confidence'] is True


def test_four_cameras_and_no_cross_camera_id_stitching():
    config,ctx=config_and_contexts(4)
    result=estimator.estimate(config,ctx,request())
    assert len(result['cameras'])==4 and result['aggregation_allowed']
    assert result['estimates'][0]['expected_arrivals']==4
    assert result['estimates'][0]['expected_center_count']==14
    assert len({r['event_namespace'] for r in result['routes']})==4
    assert all(r['confirmed_events'][0]['track_id']==1 for r in result['routes'])


@pytest.mark.parametrize('mode', ['independent','synchronized'])
def test_unsynchronized_sources_are_never_added(mode):
    config,ctx=config_and_contexts(4);config['integration_mode']=mode
    if mode=='synchronized':config['cameras'][1]['clock_verified']=False
    result=estimator.estimate(config,ctx,request(at='2026-11-17T00:00:35+09:00'))
    assert not result['aggregation_allowed']
    assert all(e['expected_center_count'] is None and e['expected_arrivals'] is None for e in result['estimates'])


def test_real_clock_alignment_uses_each_video_start_not_processing_time():
    config,ctx=config_and_contexts(4);config['integration_mode']='synchronized'
    result=estimator.estimate(config,ctx,request(at='2026-11-17T00:00:35+09:00'))
    assert [c['offset_seconds'] for c in result['cameras']]==[35,34,33,32]
    assert result['aggregation_allowed']
    next(iter(ctx.values()))['analysis_mode']='test'
    assert 'test_analysis_not_real_simultaneous_observation' in estimator.estimate(config,ctx,request(at='2026-11-17T00:00:35+09:00'))['aggregation_reasons']


@pytest.mark.parametrize('cause',['video','group','unverified','boundary'])
def test_duplicate_or_unverified_counts_cannot_be_aggregated(cause):
    config,ctx=config_and_contexts(2)
    if cause=='video':list(ctx.values())[1]['video_id']=list(ctx.values())[0]['video_id']
    if cause=='group':config['corridors'][1]['counting_group']=config['corridors'][0]['counting_group']
    if cause=='unverified':config['corridors'][1]['independent_counting_verified']=False
    if cause=='boundary':config['corridors'][1]['boundary_verified']=False
    r=estimator.estimate(config,ctx,request())
    assert not r['aggregation_allowed'] and r['estimates'][0]['expected_center_count'] is None


def test_unobserved_center_is_not_zero_but_arrivals_remain_available():
    config,ctx=config_and_contexts();config['concentration_policy']['center_observer_camera_id']=None
    result=estimator.estimate(config,ctx,request())
    assert result['observed_center_count'] is None
    assert result['estimates'][0]['expected_arrivals']==1
    assert result['estimates'][0]['expected_center_count'] is None


def test_approach_requires_arrival_assumptions_and_does_not_become_center_departure():
    config,ctx=config_and_contexts();gate=config['corridors'][0];gate['role']='approach'
    result=estimator.estimate(config,ctx,request())
    assert result['estimates'][0]['expected_arrivals'] is None
    assert result['estimates'][0]['inward_pressure_per_minute']==6
    gate.update(arrival_basis='assumption',arrival_ratio=.5,travel_seconds=5,arrival_evidence='synthetic scenario only')
    result=estimator.estimate(config,ctx,request())
    assert result['estimates'][0]['expected_arrivals']==.25
    assert result['estimates'][0]['expected_departures'] is None and result['estimates'][0]['expected_center_count'] is None
    assert result['routes'][0]['arrival_basis']=='assumption'


def test_missing_frame_unknown_id_and_insufficient_history_block_prediction():
    config,ctx=config_and_contexts()
    assert estimator.estimate(config,ctx,request(5))['estimates'][0]['expected_center_count'] is None
    source=next(iter(ctx.values()));source['replay']['frames'][25]['observed']=False
    assert not estimator.estimate(config,ctx,request())['aggregation_allowed']
    assert frame_at(source['replay'],25) is None


def test_future_observations_do_not_change_past_estimates():
    config,ctx=config_and_contexts();a=estimator.estimate(config,ctx,request())
    changed=deepcopy(ctx)
    for c in changed.values():
        c.pop('_gate_events',None)
        for f in c['replay']['frames'][31:]:f['points'][0]['x']=0;f['zones'][0]['count']=9999
    b=estimator.estimate(config,changed,request())
    assert a['estimates']==b['estimates'] and a['routes']==b['routes']


def test_approach_delay_needs_history_and_original_content_hash_duplicates_are_blocked():
    config,ctx=config_and_contexts();gate=config['corridors'][0]
    gate.update(role='approach',arrival_basis='assumption',arrival_ratio=.5,travel_seconds=40,arrival_evidence='synthetic assumption')
    result=estimator.estimate(config,ctx,request())
    assert result['estimates'][0]['expected_arrivals'] is None
    assert 'insufficient_arrival_history' in result['estimates'][0]['reasons']
    config,ctx=config_and_contexts(2)
    for c in ctx.values():c['source_sha256']='same-original-video-hash'
    result=estimator.estimate(config,ctx,request())
    assert 'duplicate_video_roi_source' in result['aggregation_reasons']
    assert 'duplicate_crossing_evidence' in result['aggregation_reasons']


def test_horizon_metrics_need_observed_future_center_and_label_synthetic_mode():
    config,ctx=config_and_contexts()
    result=evaluate_history(config,ctx,max_samples=5)
    assert [r['horizon_seconds'] for r in result['metrics']]==[10,30,60]
    assert all(r['samples']>0 and r['mae'] is not None and r['rmse'] is not None for r in result['metrics'])
    assert result['mode']=='scenario' and not result['five_minute_supported']
    config['concentration_policy']['center_observer_camera_id']=None
    assert all(r['mae'] is None and r['samples']==0 for r in evaluate_history(config,ctx)['metrics'])


def test_thresholds_are_optional_proposals_not_default_safety_verdict():
    config,ctx=config_and_contexts();r=estimator.estimate(config,ctx,request())
    assert r['state']=='미분석'
    config['concentration_policy']['attention_net_per_minute']=1
    assert estimator.estimate(config,ctx,request())['state']=='주의'
    config['zones'][1]['capacity']=5;config['concentration_policy']['urgent_capacity_ratio']=1
    result=estimator.estimate(config,ctx,request())
    assert result['state']=='긴급 확인 필요' and result['threshold_basis']=='administrator_proposal_not_safety_standard'


def test_multiple_corridors_increasing_is_scenario_only_without_synchronized_sources():
    config,ctx=config_and_contexts(4)
    r=estimator.estimate(config,ctx,request(100))
    assert r['increasing_corridors']==4 and r['state']=='주의'
    assert 'multiple_corridors_increasing' in r['state_basis']
    config['integration_mode']='independent'
    r=estimator.estimate(config,ctx,request(100))
    assert r['increasing_corridors'] is None and r['state']=='미분석'


def test_contract_rejects_venue_total_or_wrong_center():
    data=dict(model_version='team-v1',artifact_sha256='a'*64,input_columns=['count'],target_kind='venue_total',horizons_seconds=[10],training_unit='camera',camera_or_zone_identifiers=['center'],minimum_history_seconds=60,python_version='team-version',package_versions={'torch':'team-version'},preprocessing_version='v1',validation_metrics={'mae':'not-yet-reproduced'},example_output={'value':'provided by team'})
    with pytest.raises(ValueError):validate_center_output(ForecastContract(**data),dict(target_zone_id='center',horizon_seconds=10,value=10),'center')
    contract=ForecastContract(**{**data,'target_kind':'venue_zone','target_zone_id':'center'})
    out=validate_center_output(contract,dict(target_zone_id='center',horizon_seconds=10,value=10),'center')
    assert out['source']=='external_forecast_model'
    with pytest.raises(ValueError):validate_center_output(contract,dict(target_zone_id='center',horizon_seconds=300,value=10),'center')


def test_mapping_api_requires_saved_job_and_bounds(client,seeded,bundle):
    assert upload(client,seeded.id,bundle).status_code==201
    config,_=config_and_contexts(4)
    for c in config['cameras']:c.update(job_id=str(seeded.id),roi_name='near')
    config['concentration_policy']['center_observer_camera_id']=None
    venue='qa-'+uuid4().hex
    response=client.post(f'/api/venues/{venue}/revisions',json={'base_revision':0,'configuration':config})
    assert response.status_code==201,response.text
    saved=response.json()['configuration']
    assert len(saved['camera_sources'])==4
    assert saved['mapping_roi_snapshots'][str(seeded.id)]==bundle[1]
    config['corridors'][0]['boundary'][1]=[9999,9999]
    assert client.post(f'/api/venues/{venue}/revisions',json={'base_revision':1,'configuration':config}).status_code==422


def test_estimate_api_records_unknown_without_fabricating_and_operator_history(client,db):
    venue='qa-'+uuid4().hex
    response=client.post(f'/api/venues/{venue}/revisions',json={'base_revision':0,'configuration':{'name':'Actual address not supplied'}})
    assert response.status_code==201
    run=client.post(f'/api/venues/{venue}/concentration/estimate',json={'revision':1,'offset_seconds':30})
    assert run.status_code==201,run.text
    data=run.json();assert data['observed_center_count'] is None and data['registered_address'] is None and not data['aggregation_allowed']
    assert db.get(ConcentrationRun,UUID(data['run_id'])) is not None
    assert client.post(f'/api/venues/{venue}/operator-events',json={'run_id':data['run_id'],'operator':'QA','verification':'checking','action':'field_check','note':'fixture-only'}).status_code==201
    rows=client.get(f'/api/venues/{venue}/operator-events').json();assert rows[0]['verification']=='checking'
    assert client.post('/api/venues/different/operator-events',json={'run_id':data['run_id'],'operator':'QA'}).status_code==404
    assert client.post(f'/api/venues/{venue}/concentration/estimate',json={'revision':2}).status_code==409
    evaluation=client.post(f'/api/venues/{venue}/concentration/evaluate',json={'revision':1})
    assert evaluation.status_code==200 and all(m['mae'] is None for m in evaluation.json()['metrics'])
    assert client.get('/api/venues/forecast-contract/schema').json()['status']=='awaiting_team_artifact_no_model_loaded'


def test_configuration_rejects_clock_without_timezone_or_evidence_and_wrong_center():
    config,_=config_and_contexts()
    config['cameras'][0]['capture_start']='2026-11-17T00:00:00'
    with pytest.raises(ValidationError):Configuration(**config)
    config,_=config_and_contexts();config['corridors'][0]['to_zone_id']='absent'
    with pytest.raises(ValidationError):Configuration(**config)
