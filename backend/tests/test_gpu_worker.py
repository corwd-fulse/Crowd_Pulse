import csv
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import worker as module
from app.models import AnalysisJob, VisionResult
from app.services.inference_runtime import normalize_device, check_device, predictor_settings, InferenceFailure
from test_worker import worker as worker, seeded as seeded, bundle as bundle


@pytest.mark.parametrize('value,expected',[('cpu','cpu'),('0','cuda:0'),('cuda','cuda:0'),('cuda:0','cuda:0'),('2','cuda:2')])
def test_device_normalization(value,expected):
    assert normalize_device(value)==expected


@pytest.mark.parametrize('value',['auto','-1','0,1','cuda:-1','mps'])
def test_invalid_device(value):
    with pytest.raises(ValueError):normalize_device(value)


def test_cuda_availability_and_index_without_hardware():
    fake=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False,device_count=lambda:0))
    with pytest.raises(InferenceFailure,match='CUDA_UNAVAILABLE'):check_device('0',fake)
    fake.cuda.is_available=lambda:True
    fake.cuda.device_count=lambda:1
    assert check_device('cuda:0',fake)=='cuda:0'
    with pytest.raises(InferenceFailure,match='CUDA_DEVICE_UNAVAILABLE'):check_device('cuda:1',fake)


def test_cpu_defaults_and_option_validation(worker):
    assert worker.device=='cpu' and worker.imgsz==1920
    with pytest.raises(ValueError):module.Worker(worker.model_path,'v',imgsz=641)


def test_backend_model_is_verified_instead_of_cpu_wrapper():
    import torch
    parameter=SimpleNamespace(device=torch.device('cuda:0'),dtype=torch.float32,is_floating_point=lambda:True)
    backend=SimpleNamespace(device=torch.device('cuda:0'),model=SimpleNamespace(parameters=lambda:iter([parameter])))
    details=predictor_settings(SimpleNamespace(device=torch.device('cuda:0'),model=backend),'cuda:0')
    assert details['device']=='cuda:0' and details['device_verified']
    backend.device=torch.device('cpu')
    with pytest.raises(InferenceFailure,match='DEVICE_MISMATCH'):
        predictor_settings(SimpleNamespace(device=torch.device('cuda:0'),model=backend),'cuda:0')


def test_worker_gpu_unavailable_is_failed_without_tracking(worker,seeded,db,monkeypatch):
    worker.device='cuda:0'
    def unavailable(device):raise InferenceFailure('CUDA_UNAVAILABLE: test environment')
    monkeypatch.setattr(module,'check_device',unavailable)
    with worker.exclusive():result=worker.run_once()
    assert result['status']=='failed' and result['error'].startswith('CUDA_UNAVAILABLE')
    worker.track.assert_not_called()
    db.expire_all()
    assert db.get(AnalysisJob,seeded.id).status=='failed'
    assert db.get(VisionResult,seeded.id) is None


def test_gpu_oom_records_clear_error_and_does_not_retry(worker,seeded,db,monkeypatch):
    worker.device='cuda:0'
    monkeypatch.setattr(module,'check_device',lambda device:device)
    worker.track.side_effect=RuntimeError('CUDA out of memory. simulated')
    with worker.exclusive():
        result=worker.run_once()
        assert worker.run_once() is None
    assert result['status']=='failed' and result['error'].startswith('CUDA_OUT_OF_MEMORY')
    worker.track.assert_called_once()
    db.expire_all()
    assert db.get(VisionResult,seeded.id) is None
    assert db.get(AnalysisJob,seeded.id).error_message.startswith('CUDA_OUT_OF_MEMORY')


def test_real_adapter_forwards_gpu_options_and_saves_runtime_metadata(worker,seeded,db,monkeypatch,tmp_path):
    monkeypatch.setenv('YOLO_CONFIG_DIR',str(tmp_path/'config'))
    from app.services import tracker
    worker.device='cuda:0';worker.imgsz=960
    original=worker.track
    def tracking(source,out,name,**options):
        assert options=={'model_path':worker.model_path,'imgsz':960,'device':'cuda:0'}
        result=original(source,out,name)
        result['execution_settings']={'device':'cuda:0','requested_device':'cuda:0','predictor_device':'cuda:0',
            'device_verified':True,'imgsz':960,'batch':1,'half':False,'torch_version':'2.5.1+cu118',
            'cuda_version':'11.8','model_name':worker.model_path.name}
        return result
    spy=Mock(side_effect=tracking)
    monkeypatch.setattr(tracker,'track_video',spy)
    monkeypatch.setattr(module,'check_device',lambda device:device)
    worker.track=None
    with worker.exclusive():result=worker.run_once()
    assert result['status']=='succeeded',result
    spy.assert_called_once()
    db.expire_all()
    saved=db.get(VisionResult,seeded.id).execution_snapshot['settings']
    assert saved['device']=='cuda:0' and saved['imgsz']==960
    assert saved['torch_version']=='2.5.1+cu118' and saved['cuda_version']=='11.8'
    assert saved['batch']==1 and saved['half'] is False and len(saved['model_sha256'])==64


@pytest.mark.parametrize('size,explicit',[(1920,False),(960,True)])
def test_tracker_settings_csv_compatibility_and_callback_cleanup(tmp_path,monkeypatch,size,explicit):
    monkeypatch.setenv('YOLO_CONFIG_DIR',str(tmp_path/'config'))
    import torch
    from app.services import tracker
    model=SimpleNamespace(callbacks={'on_predict_start':[]})
    model.add_callback=lambda event,callback:model.callbacks[event].append(callback)
    predictor=SimpleNamespace(device=torch.device('cpu'),model=SimpleNamespace(device=torch.device('cpu'),model=torch.nn.Linear(2,2)))
    boxes=SimpleNamespace(data=torch.zeros(1,7),id=torch.tensor([7]),xyxy=torch.tensor([[10.,20.,30.,80.]]),conf=torch.tensor([.9]))
    class Boxes:
        data=boxes.data;id=boxes.id;xyxy=boxes.xyxy;conf=boxes.conf
        def __len__(self):return 1
    received={}
    def track(*args,**options):
        received.update(options)
        for callback in model.callbacks['on_predict_start']:callback(predictor)
        yield SimpleNamespace(boxes=Boxes())
    model.track=track
    monkeypatch.setattr(tracker,'_get_model',lambda path:model)
    cap=Mock();cap.get.return_value=25
    monkeypatch.setattr(tracker.cv2,'VideoCapture',lambda _:cap)
    options={'model_path':tmp_path/'model.pt','imgsz':size,'device':'cpu'} if explicit else {}
    summary=tracker.track_video(tmp_path/'video.mp4',tmp_path/'out','video',**options)
    assert received['imgsz']==size and received['device']=='cpu' and received['half'] is False and received['batch']==1
    with open(summary['detections_csv']) as f:
        rows=list(csv.reader(f))
    assert rows[0]==['frame','t_ms','track_id','x1','y1','x2','y2','conf']
    assert rows[1][:3]==['0','0','7']
    assert summary['execution_settings']['device_verified']
    assert model.callbacks['on_predict_start']==[]
    assert not predictor.model.model._forward_pre_hooks


def test_cpu_and_gpu_workers_share_single_exclusive_slot(worker):
    from app.worker import WorkerBusy
    other=module.Worker(worker.model_path,'gpu-test',device='cuda:0',imgsz=960,base_dir=worker.base)
    with worker.exclusive():
        with pytest.raises(WorkerBusy):
            with other.exclusive():pass


def test_cli_gpu_options_and_stop_after_oom(tmp_path,monkeypatch):
    import sys
    from contextlib import nullcontext
    weights=tmp_path/'local.pt';weights.write_bytes(b'fixture')
    fake=SimpleNamespace(exclusive=lambda:nullcontext(),recover_interrupted=lambda:0,
                         run_once=Mock(return_value={'status':'failed','error':'CUDA_OUT_OF_MEMORY: simulated'}))
    constructor=Mock(return_value=fake)
    monkeypatch.setattr(module,'Worker',constructor)
    monkeypatch.setattr(module.signal,'signal',Mock())
    monkeypatch.setattr(sys,'argv',['worker','--model-path',str(weights),'--algorithm-version','test','--device','0','--imgsz','960'])
    assert module.main()==1
    assert constructor.call_args.kwargs['device']=='cuda:0'
    assert constructor.call_args.kwargs['imgsz']==960
    fake.run_once.assert_called_once()
