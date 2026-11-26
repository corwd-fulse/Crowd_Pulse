import json

import pandas as pd
import pytest

from app.services import job_forecast as jf


def _out(tmp_path, residual=True, dmcount=True):
    out = tmp_path / 'run'
    (out / 'pipeline').mkdir(parents=True)
    (out / 'roi_snapshot.json').write_text(json.dumps({'zones': [{'name': 'A', 'polygon': [[0, 0], [100, 0], [100, 50], [0, 50]]}]}), encoding='utf-8')
    secs = range(25)
    pd.DataFrame({'sec': secs, 'zone': 'A', 'count': [10.0] * 25, 'inflow': 0, 'outflow': 0}).to_csv(out / 'pipeline/crowd_observations.csv', index=False)
    pd.DataFrame({'sec': secs, 'zone': 'A', 'angle_deg': [92.0] * 20 + [10.0] * 5, 'magnitude_px': 1.0, 'motion': 1.0,
                  'low_confidence': [False] * 25}).to_csv(out / 'pipeline/flow.csv', index=False)
    for n in ('individual_directions', 'bottleneck_scores'):
        (out / f'pipeline/{n}.csv').write_text('x\n1\n', encoding='utf-8')
    if dmcount:
        (out / 'extended').mkdir()
        dm = pd.DataFrame({'window_start': [0, 10], 'zone': 'A', 'estimated_count': [3.0, 5.0]})
        if residual:
            dm['yolo_residual_count'] = [1.5, 0.5]
        dm.to_csv(out / 'extended/dmcount_estimates.csv', index=False)
    return out


def test_prepare_builds_three_sources_with_window_values(tmp_path):
    work, sources = jf.prepare(_out(tmp_path))
    assert sources == ['yolo', 'dmcount', 'fused']
    dm = pd.read_csv(work / 'dmcount/input/crowd_observations.csv').set_index('sec')['count']
    fu = pd.read_csv(work / 'fused/input/crowd_observations.csv').set_index('sec')['count']
    assert dm[0] == 3.0 and dm[15] == 5.0 and dm[24] == 5.0   # 20~24초 창이 없으면 앞 창 값
    assert fu[5] == 11.5 and fu[12] == 10.5 and fu[22] == 10.5
    yolo = pd.read_csv(work / 'yolo/input/crowd_observations.csv')
    assert (yolo['count'] == 10.0).all()


def test_zone_meta_is_marked_assumed(tmp_path):
    work, _ = jf.prepare(_out(tmp_path))
    meta = json.loads((work / 'zone_meta.json').read_text(encoding='utf-8'))['A']
    assert meta['area_m2'] == 2.0 and meta['width_m'] == 2.0      # 100x50 px / 50 px/m
    assert meta['exit_angle_deg'] == 95.0 and meta['area_source'] == 'assumed'


def test_prepare_without_dmcount_or_residual(tmp_path):
    assert jf.prepare(_out(tmp_path / 'a', dmcount=False))[1] == ['yolo']
    assert jf.prepare(_out(tmp_path / 'b', residual=False))[1] == ['yolo', 'dmcount']


def test_missing_outputs_skip_cleanly(tmp_path):
    out = _out(tmp_path)
    (out / 'pipeline/flow.csv').unlink()
    with pytest.raises(jf.ForecastSkipped):
        jf.prepare(out)
    with pytest.raises(jf.ForecastSkipped):
        jf.run_for_job(None, None, out, repo_root=tmp_path)   # forecast 패키지 없음
