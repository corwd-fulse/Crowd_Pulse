from copy import deepcopy
from uuid import UUID
import pytest
from sqlalchemy import select
from app.services.zone_replay import build_replay
from app.services.roi_geometry import validate_disjoint
from app.models import Video,VideoRoiRevision,AnalysisJob
from test_vision_results import seeded,bundle,client
from test_worker import worker

ZONES=[dict(name=n,polygon=[[x,0],[x+100,0],[x+100,100],[x,100]]) for n,x in [('A',0),('B',120),('C',240)]]
def replay(xs,*,gaps=(),many=1,zones=None):
    data=dict(analysis_job_id='job',video_id='video',fps=25,frame_count=len(xs),video_resolution=[360,120],frames=[
        dict(frame=f,t_ms=f*40,boxes=[dict(track_id=tid,x1=x-2,x2=x+2,y1=20,y2=50,confidence=.9) for tid in range(1,many+1)])
        for f,x in enumerate(xs) if f not in gaps])
    return build_replay(data,dict(zones=zones or ZONES,video_resolution=[360,120]),observed_frames=set(range(len(xs))))
def routes(result):return {(r['from_zone'],r['to_zone']):r['transitions'] for r in result['routes']}

def test_bidirectional_and_reentry_keep_events_distinct_from_od():
    r=replay(list(range(30,181,10))+list(range(170,29,-10)))
    assert routes(r)=={('A','B'):1,('B','A'):1}
    assert r['observed_od']==[dict(from_zone='A',to_zone='A',tracks=1)]
    assert len({e['event_id'] for e in r['events']})==len(r['events'])

def test_a_b_c_and_many_people():
    r=replay(list(range(30,301,10)),many=4)
    assert routes(r)=={('A','B'):4,('B','C'):4}
    assert r['observed_od']==[dict(from_zone='A',to_zone='C',tracks=4)]
    end=r['frames'][-1]['zones']
    assert next(z for z in end if z['zone']=='B')['inflow']==4
    assert next(z for z in end if z['zone']=='B')['outflow']==4

def test_gap_never_creates_a_path_or_od():
    r=replay(list(range(30,181,10)),gaps=(7,8,9,10))
    assert routes(r)=={} and r['observed_od']==[]
    assert any(d['reason']=='id_gap_or_duplicate' for d in r['diagnostics'])
    assert r['frames'][7]['zones'][0]['inflow'] is None

def test_hysteresis_and_boundary_jitter_do_not_count_transitions():
    r=replay([70,80,90]+[99,101,98,102]*5)
    assert not r['events']

def test_id_jump_and_overlap_are_unknown_not_crossings():
    r=replay([30,31,32,180,181,182,183])
    assert not r['routes']
    assert any(d['reason']=='suspected_id_switch' for d in r['diagnostics'])
    overlapping=[dict(name='A',polygon=[[0,0],[100,0],[100,100],[0,100]]),dict(name='B',polygon=[[50,0],[150,0],[150,100],[50,100]])]
    with pytest.raises(ValueError):validate_disjoint(overlapping)
    assert replay([60]*6,zones=overlapping)['frames'][0]['zones'][0]['count'] is None

def test_unobserved_frame_is_not_zero_people():
    d=dict(analysis_job_id='j',video_id='v',fps=25,frame_count=1,video_resolution=[360,120],frames=[dict(frame=0,t_ms=0,boxes=[])])
    r=build_replay(d,dict(zones=ZONES))
    assert r['frames'][0]['zones'][0]['count'] is None


def test_observed_outside_path_counts_route_once_and_boundary_counts_once():
    xs=list(range(30,96,5))+[105,110,115,110,115,110]+list(range(125,171,5))
    r=replay(xs)
    assert routes(r)=={('A','B'):1}
    assert any(e['kind']=='zone_path' and e['via_outside'] for e in r['events'])
    final=r['frames'][-1]['zones']
    assert next(z for z in final if z['zone']=='A')['outflow']==1
    assert next(z for z in final if z['zone']=='B')['inflow']==1


def test_exit_and_reentry_to_same_zone_is_not_a_fabricated_a_b_route():
    xs=list(range(30,96,5))+[105,110,115,110,115,110]+list(range(95,29,-5))
    r=replay(xs)
    assert routes(r)=={}
    a=r['frames'][-1]['zones'][0]
    assert a['inflow']==1 and a['outflow']==1
    assert r['observed_od']==[dict(from_zone='A',to_zone='A',tracks=1)]


@pytest.mark.integration
def test_roi_revisions_are_append_only_and_jobs_hold_snapshot(db,seeded,bundle,client,monkeypatch):
    from routers import video_rois,analysis
    monkeypatch.setattr(video_rois,'video_resolution',lambda video:[1920,1080])
    monkeypatch.setattr(analysis.settings,'ALLOW_TEST_ANALYSIS',True)
    video=db.get(Video,seeded.video_id);original=deepcopy(bundle[1]);video.roi_snapshot=deepcopy(original)
    seeded.status='cancelled';db.flush()
    url=f'/api/videos/{video.id}'
    current=client.get(url+'/roi').json()
    changed=deepcopy(original);changed['zones'][0].update(name='A',color='#23cfff',visible=False)
    result=client.post(url+'/roi-revisions',json=changed,headers={'If-Match':current['fingerprint']})
    assert result.status_code==201,result.text
    assert db.get(Video,video.id).roi_snapshot==original
    assert client.post(url+'/roi-revisions',json=changed,headers={'If-Match':current['fingerprint']}).status_code==412
    job=client.post('/api/analysis/jobs',json=dict(video_id=str(video.id),analysis_mode='test')).json()
    snapshot=deepcopy(db.get(AnalysisJob,UUID(job['id'])).roi_snapshot)
    changed['zones'][0]['name']='B'
    assert client.post(url+'/roi-revisions',json=changed,headers={'If-Match':result.json()['fingerprint']}).status_code==201
    assert db.get(AnalysisJob,UUID(job['id'])).roi_snapshot==snapshot
    assert len(db.scalars(select(VideoRoiRevision).where(VideoRoiRevision.video_id==video.id)).all())==2


@pytest.mark.integration
def test_revision_rejects_mismatch_and_overlapping_geometry(db,seeded,bundle,client,monkeypatch):
    from routers import video_rois
    monkeypatch.setattr(video_rois,'video_resolution',lambda video:[1920,1080])
    video=db.get(Video,seeded.video_id);video.roi_snapshot=deepcopy(bundle[1]);db.flush()
    url=f'/api/videos/{video.id}'
    fingerprint=client.get(url+'/roi').json()['fingerprint']
    invalid=deepcopy(bundle[1]);invalid['video_resolution']=[100,100]
    assert client.post(url+'/roi-revisions',json=invalid,headers={'If-Match':fingerprint}).status_code==422
    invalid=deepcopy(bundle[1]);invalid['zones']+=[dict(name='B',polygon=invalid['zones'][0]['polygon'])]
    assert client.post(url+'/roi-revisions',json=invalid,headers={'If-Match':fingerprint}).status_code==422


@pytest.mark.integration
def test_replay_api_validates_frame_index_and_never_writes_results(db,seeded,bundle,client):
    from uuid import uuid4
    from app.services import vision_results as storage
    from app.models import VisionResult
    paths,roi,meta=bundle
    result,_=storage.store_result(db,seeded.id,paths,roi,meta)
    token=uuid4();seeded.worker_token=token;db.flush()
    folder=storage.BASE_DIR/'uploads/analysis'/str(seeded.id)/(token.hex+'-fixture')
    tracker=folder/'tracker';tracker.mkdir(parents=True)
    (folder/'execution_metadata.json').write_text(meta.model_dump_json(),encoding='utf8')
    (tracker/'video_detections.csv').write_text('frame,t_ms,track_id,x1,y1,x2,y2,conf\n1,40,1,10,10,20,40,0.9\n',encoding='utf8')
    index=tracker/'video_frames.csv'
    index.write_text('frame,t_ms,count\n'+''.join(f'{f},{f*40},{int(f==1)}\n' for f in range(404)),encoding='utf8')
    snapshot=deepcopy(result.execution_snapshot)
    response=client.get(f'/api/analysis/jobs/{seeded.id}/replay')
    assert response.status_code==200,response.text
    assert response.json()['frames'][0]['zones'][0]['count']==0
    assert response.json()['frames'][1]['zones'][0]['count']==1
    index.write_text('frame,t_ms,count\n0,0,9\n',encoding='utf8')
    assert client.get(f'/api/analysis/jobs/{seeded.id}/replay').status_code==422
    db.expire_all();assert db.get(VisionResult,seeded.id).execution_snapshot==snapshot


@pytest.mark.integration
def test_worker_persists_new_transition_artifact_metadata(worker,seeded,db):
    import json
    from pathlib import Path
    from app.models import VisionResult
    summary=dict(version='observed-zone-transitions-v1',events=[],routes=[],observed_od=[])
    original=worker.pipeline.side_effect
    def pipeline(*args):
        paths=original(*args)
        (Path(args[3])/'analysis_details.json').write_text(json.dumps(dict(transitions=summary)),encoding='utf8')
        return paths
    worker.pipeline.side_effect=pipeline
    with worker.exclusive():result=worker.run_once()
    assert result['status']=='succeeded'
    db.expire_all()
    assert db.get(VisionResult,seeded.id).execution_snapshot['settings']['transitions']==summary
