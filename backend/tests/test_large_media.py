import csv
import json
from pathlib import Path
import pandas as pd
import pytest
from app.services import replay_index
from app.services.zone_replay import build_replay
from app.services.streaming_tracks import process
from app.services.zone_analysis import Zone,compute_zone_timeseries
from app.services.individual_flow import compute_individual_directions,aggregate_by_zone
from app.services.hybrid_flow import build_hybrid_summary
from test_vision_results import bundle,seeded,client,upload

def test_streamed_import_merges_each_metric_kind_without_overwriting_other_columns(bundle,seeded,client,db,monkeypatch):
    from app.services import vision_csv
    from app.models import VisionZoneMetric
    from sqlalchemy import select
    monkeypatch.setattr(vision_csv,'MEMORY_ROWS',1)
    response=upload(client,seeded.id,bundle)
    assert response.status_code==201,response.text
    rows=db.scalars(select(VisionZoneMetric).where(VisionZoneMetric.analysis_job_id==seeded.id).order_by(VisionZoneMetric.video_offset_ms)).all()
    assert len(rows)==2
    assert float(rows[0].mean_person_count)==3.9 and rows[0].has_observation
    assert rows[0].has_flow and rows[0].motion==.04
    assert rows[0].has_individual_aggregate and rows[0].direction_sample_count==16
    assert rows[1].has_flow and rows[1].motion is None and rows[1].low_confidence

def test_duplicate_after_streaming_boundary_is_rejected(bundle,monkeypatch):
    from app.services import vision_csv
    monkeypatch.setattr(vision_csv,'MEMORY_ROWS',1)
    paths,roi,metadata=bundle
    with paths['crowd_observations'].open('a') as stream:stream.write('0,near,3.9,1,2\n')
    with pytest.raises(vision_csv.InvalidResult,match='row 4'):vision_csv.validate_bundle(paths,roi,metadata)

def test_replay_api_distinguishes_pending_invalid_range_and_dense_page(seeded,client,monkeypatch):
    def pending(*_):raise replay_index.IndexPending()
    monkeypatch.setattr(replay_index,'ensure',pending)
    url=f'/api/analysis/jobs/{seeded.id}/replay'
    response=client.get(url+'?start_ms=0&end_ms=5000')
    assert response.status_code==202 and response.headers['retry-after']=='1'
    assert response.json()['status']=='indexing'
    assert client.get(url+'?start_ms=0&end_ms=31000').status_code==422
    monkeypatch.setattr(replay_index,'ensure',lambda *_:Path('not-opened'))
    def dense(*_,**__):raise replay_index.WindowTooDense('No observations were sampled')
    monkeypatch.setattr(replay_index,'window',dense)
    assert client.get(url+'?start_ms=0&end_ms=5000').status_code==413
    assert client.get(url+'?start_ms=33.3667&end_ms=5033.3667').status_code==413

def test_h264_copy_preserves_original_and_every_frame(tmp_path):
    import cv2,numpy as np
    from app.services import playback_media
    try:playback_media.binary('ffmpeg')
    except ValueError:pytest.skip('Optional portable FFmpeg not configured')
    path=tmp_path/'source.mp4';target=tmp_path/'playback.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25,(160,96))
    for i in range(75):writer.write(np.full((96,160,3),i,dtype=np.uint8))
    writer.release();before=playback_media.sha(path)
    result=playback_media.convert(path,target)
    assert result['frame_count']==75 and result['verified_every_frame']
    assert result['max_frame_timestamp_error_seconds']==0
    assert playback_media.sha(path)==before==result['source_sha256']

def source(tmp_path):
    path=tmp_path/'video_detections.csv';frames=[]
    # Out-of-order rows, long ID, outside passage, gap and jump.
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['frame','t_ms','track_id','x1','y1','x2','y2','conf'])
        for f in range(320):
            boxes=[]
            for tid in (3,1,2):
                if tid==2 and 140<=f<145:continue
                x=30+min(f,319-f)*2+tid
                if tid==3 and f>=200:x=50
                writer.writerow([f,f*40,tid,x-2,10,x+2,50,.9])
                boxes.append(dict(track_id=tid,x1=x-2,y1=10,x2=x+2,y2=50,confidence=.9))
            frames.append(dict(frame=f,t_ms=f*40,boxes=boxes))
    fp=tmp_path/'video_frames.csv'
    with fp.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['frame','t_ms','count']);writer.writerows((f['frame'],f['t_ms'],len(f['boxes'])) for f in frames)
    zones=[Zone('A',[(0,0),(100,0),(100,100),(0,100)]),Zone('B',[(120,0),(360,0),(360,100),(120,100)])]
    roi=dict(video_resolution=[360,120],zones=[dict(name=z.name,polygon=z.polygon) for z in zones])
    meta=dict(version='observed-zone-transitions-v1',analysis_job_id='j',video_id='v',fps=25.,frame_count=320,video_resolution=[360,120],roi_snapshot=roi)
    return path,fp,frames,zones,meta

def test_disk_window_preserves_events_counts_and_page_boundary(tmp_path):
    path,fp,frames,zones,meta=source(tmp_path);destination=tmp_path/'index.sqlite'
    replay_index.build(path,fp,meta,destination,(path.stat().st_size,path.stat().st_mtime_ns),(fp.stat().st_size,fp.stat().st_mtime_ns))
    legacy=build_replay({**meta,'frames':frames},meta['roi_snapshot'],observed_frames=range(320))
    for start in (0,5000,10000,12320):
        page=replay_index.window(destination,start,min(start+5000,12800))
        assert page['routes']==legacy['routes'];assert page['coverage']==legacy['coverage']
        assert page['observed_od']==legacy['observed_od']
        for frame in page['frames']:assert frame==legacy['frames'][frame['frame']]
        assert page['events']==[e for e in legacy['events'] if e['frame']<page['window']['end_ms']/40]
    with pytest.raises(ValueError):replay_index.window(destination,0,31000)

def test_dense_window_shortens_only_at_complete_frame_boundaries(tmp_path,monkeypatch):
    path,fp,frames,zones,meta=source(tmp_path);destination=tmp_path/'index.sqlite'
    replay_index.build(path,fp,meta,destination,(path.stat().st_size,path.stat().st_mtime_ns),(fp.stat().st_size,fp.stat().st_mtime_ns))
    monkeypatch.setattr(replay_index,'MAX_RESPONSE_POINTS',18)
    page=replay_index.window(destination,0,5000)
    assert len(page['frames'])==6 and page['window']['next_start_ms']==240
    next_page=replay_index.window(destination,240,5240)
    assert next_page['frames'][0]['frame']==5 # heading context, not lost ID history
    assert all(len(f['points'])==3 for f in page['frames']+next_page['frames'])
    monkeypatch.setattr(replay_index,'MAX_RESPONSE_POINTS',2)
    with pytest.raises(replay_index.WindowTooDense):replay_index.window(destination,0,5000)
    monkeypatch.setattr(replay_index,'MAX_RESPONSE_POINTS',3)
    with pytest.raises(replay_index.WindowTooDense):replay_index.window(destination,240,5240)

def test_lazy_multi_camera_context_keeps_gate_and_quality_policy(tmp_path,monkeypatch):
    from app.services import indexed_context
    from app.services.concentration import gate_events,frame_at,window_quality
    path,fp,frames,zones,meta=source(tmp_path);destination=tmp_path/'index.sqlite'
    replay_index.build(path,fp,meta,destination,(path.stat().st_size,path.stat().st_mtime_ns),(fp.stat().st_size,fp.stat().st_mtime_ns))
    monkeypatch.setattr(indexed_context,'ensure',lambda *_:destination)
    indexed=indexed_context.for_job(None,None)
    legacy=build_replay({**meta,'frames':frames},meta['roi_snapshot'],observed_frames=range(320))
    gate=dict(boundary=[[110,0],[110,100]],center_side=[150,50])
    assert gate_events(indexed,gate)==gate_events(legacy,gate)
    for offset in (0,5,10,12.76):
        assert frame_at(indexed,offset)['zones'][0]['count']==frame_at(legacy,offset)['zones'][0]['count']
        assert window_quality(indexed,offset,2)==window_quality(legacy,offset,2)

def test_streamed_processing_matches_legacy_policy_and_training_values(tmp_path):
    path,fp,frames,zones,meta=source(tmp_path)
    old=compute_individual_directions(str(path));obs=compute_zone_timeseries(str(path),zones,25)
    agg=aggregate_by_zone(old,str(path),zones,25)
    motion=pd.DataFrame(columns=['sec','zone','angle_deg','magnitude_px','motion','low_confidence'])
    hybrid=build_hybrid_summary(old,str(path),zones,motion,obs)
    output=tmp_path/'individual.csv'
    with process(path,zones,25,output) as (new_obs,new_agg,votes):
        pd.testing.assert_frame_equal(pd.read_csv(output),old,check_dtype=False,atol=1e-10)
        pd.testing.assert_frame_equal(new_obs,obs,check_dtype=False,atol=1e-10)
        pd.testing.assert_frame_equal(new_agg,agg,check_dtype=False,atol=1e-10)
        new_hybrid=build_hybrid_summary(None,str(path),zones,motion,new_obs,prepared_votes=votes)
        for new,previous in zip(new_hybrid['rows'],hybrid['rows']):
            for key,value in previous.items():
                if isinstance(value,float):assert new[key]==pytest.approx(value,abs=1e-10)
                else:assert new[key]==value
