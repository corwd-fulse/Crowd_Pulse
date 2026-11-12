from datetime import datetime,timedelta,timezone

from app.models import AnalysisJob
from test_vision_results import client as client,seeded as seeded


def test_latest_completion_is_found_beyond_recent_created_jobs(client, seeded, db):
    now=datetime.now(timezone.utc)
    seeded.status='succeeded'
    seeded.created_at=now-timedelta(days=3)
    seeded.started_at=now-timedelta(hours=3)
    seeded.finished_at=now-timedelta(hours=1)
    newer_created=AnalysisJob(video_id=seeded.video_id,status='succeeded',created_at=now-timedelta(days=1),started_at=now-timedelta(hours=4),finished_at=now-timedelta(hours=2))
    db.add(newer_created)
    for i in range(101):
        db.add(AnalysisJob(video_id=seeded.video_id,status='failed',created_at=now-timedelta(minutes=i)))
    db.flush()
    base=f'/api/analysis/jobs?video_id={seeded.video_id}'
    recent=client.get(base).json()
    assert len(recent)==100 and str(seeded.id) not in [j['id'] for j in recent]
    completed=client.get(base+'&status=succeeded&order_by=finished_at&limit=1').json()
    assert len(completed)==1 and completed[0]['id']==str(seeded.id)
    assert client.get(base+'&status=queued').json()==[]
    # No list query changes any job's state.
    assert db.get(AnalysisJob,seeded.id).status=='succeeded'


def test_job_filter_validation_and_explicit_queued_detail(client, seeded):
    for query in ['status=unknown','order_by=unknown','limit=0','limit=101','offset=-1']:
        assert client.get('/api/analysis/jobs?'+query).status_code==422
    response=client.get(f'/api/analysis/jobs/{seeded.id}')
    assert response.status_code==200 and response.json()['status']=='queued'


def test_job_readiness_refers_to_request_snapshot_not_current_video_roi(client,seeded,db):
    from app.models import Video
    roi={'status':'candidate','approved_for_analysis':False,'video_resolution':[1280,720],
         'zones':[{'name':'near','polygon':[[10,10],[100,10],[100,100]]}]}
    db.get(Video,seeded.video_id).roi_snapshot=roi
    db.flush()
    url=f'/api/analysis/jobs/{seeded.id}'
    assert client.get(url).json()['roi_configured'] is False
    seeded.roi_snapshot=roi
    db.flush()
    response=client.get(url).json()
    assert response['roi_configured'] is True
    assert 'roi_snapshot' not in response


def test_capabilities_is_read_only_and_matches_server_policy(client):
    from app.database import settings
    assert client.get('/api/analysis/capabilities').json()=={
        'test_analysis_enabled':settings.ALLOW_TEST_ANALYSIS,'execution':'external_worker','auto_start_worker':False,
        'analysis_profiles':['fast','full']}
