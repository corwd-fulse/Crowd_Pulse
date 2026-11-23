import pandas as pd

from app.services.queue_analysis import compute_queue_scores


def _timeseries(sec_count_motion):
    rows = [{"sec": sec, "zone": "A", "count": count, "motion": motion} for sec, count, motion in sec_count_motion]
    return pd.DataFrame(rows)


def test_crowded_and_still_scores_higher_than_crowded_and_flowing():
    queueing = [(s, 20, 0.5) for s in range(0, 10)]  # 사람 많고 거의 안 움직임
    flowing = [(s, 20, 10.0) for s in range(10, 20)]  # 사람 많지만 잘 흐름
    ts = _timeseries(queueing + flowing)

    scores = compute_queue_scores(ts, window_s=10)
    queue_score = scores[scores.window_start == 0]["queue_score"].iloc[0]
    flow_score = scores[scores.window_start == 10]["queue_score"].iloc[0]
    assert queue_score > flow_score
    assert scores[scores.window_start == 0]["is_queue"].iloc[0]
    assert not scores[scores.window_start == 10]["is_queue"].iloc[0]


def test_empty_zone_is_not_a_queue_even_if_still():
    ts = _timeseries([(s, 0, 0.0) for s in range(0, 10)] + [(s, 20, 10.0) for s in range(10, 20)])
    scores = compute_queue_scores(ts, window_s=10)
    assert not scores[scores.window_start == 0]["is_queue"].iloc[0]
