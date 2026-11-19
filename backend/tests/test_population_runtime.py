import json, queue
from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services import population_forecast as service


def test_missing_runtime_is_explicit_and_does_not_start_worker(monkeypatch):
    monkeypatch.setenv('CROWD_FORECAST_PYTHON','C:/missing/forecast-python.exe')
    response=TestClient(app).get('/api/population-forecast/catalog')
    assert response.status_code==200
    assert response.json()['status']=='unavailable'
    assert 'lightgbm' in response.json()['missing_dependencies']


def test_invalid_history_is_rejected_before_model_work():
    client=TestClient(app)
    row=dict(observed_at='2026-10-01T00:00:00+09:00',area_id='11110515',estimate=1)
    assert client.post('/api/population-forecast/predict',json={'observations':[row,row]}).status_code==422
    assert client.post('/api/population-forecast/predict',json={'observations':[{**row,'estimate':-1}]}).status_code==422
    assert client.post('/api/population-forecast/upload',files={'file':('roi.csv',b'sec,zone,count\n0,A,1\n')}).status_code==422


@pytest.mark.parametrize('kind,code,status',[('timeout','timeout',504),('eof','process_failed',503),('bad_json','process_failed',503)])
def test_broker_failure_and_timeout_only_control_its_private_child(kind,code,status,monkeypatch):
    broker=service.ForecastBroker()
    child=Mock();child.poll.return_value=None
    broker.process=child;broker.signature=('fixture','hash');broker.output=Mock()
    if kind=='timeout':broker.output.get.side_effect=queue.Empty
    else:broker.output.get.return_value=None if kind=='eof' else 'invalid json'
    try:
        with pytest.raises(service.ForecastError) as exc:broker._run(b'input','.csv',broker.signature)
        assert exc.value.code==code and exc.value.http_status==status
        assert broker.model_state=='failed'
        child.terminate.assert_called_once()
    finally:broker.close()
