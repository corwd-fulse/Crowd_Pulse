"""Independent track/area flow comparison; never mixes their magnitudes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .zone_analysis import OUTSIDE, _bottom_center, _zone_of_point
from .individual_flow import DEFAULT_WINDOW_FRAMES

HYBRID_VERSION = 'independent-directions-v1'
MIN_TRACKS = 3
MIN_COHERENCE = 0.3
MAX_DIFFERENCE_DEG = 45.0


def circular_difference(a: float, b: float) -> float:
    return float(abs((a - b + 180) % 360 - 180))


def build_hybrid_summary(individual, detections_csv, zones, motion, observations, *, prepared_votes=None):
    """One direction vote per distinct person/second; speed uses actual elapsed time.

    Legacy speed_px remains displacement over a variable observation window.
    The JSON summary is additive execution metadata, outside CSV schema v1.
    """
    if prepared_votes is None:
        det = pd.read_csv(detections_csv)
        det['zone'] = [_zone_of_point(_bottom_center(r.x1, r.y1, r.x2, r.y2), zones)
                       for r in det.itertuples()]
        det['sec'] = (det.t_ms // 1000).astype(int)
        elapsed = {}
        for tid, group in det.sort_values('frame').groupby('track_id'):
            records = list(group.itertuples())
            for i, row in enumerate(records):
                lo = max(0, i - DEFAULT_WINDOW_FRAMES)
                dt = (row.t_ms - records[lo].t_ms) / 1000
                if dt > 0:
                    elapsed[(row.frame, tid)] = dt
        merged = individual.merge(det[['frame', 'track_id', 'zone', 'sec']],
                                  on=['frame', 'track_id'], how='left')
        valid = merged[merged.reliable & merged.angle_deg.notna() & merged.zone.notna()
                       & (merged.zone != OUTSIDE)].copy()
        valid['speed_px_per_second'] = [r.speed_px / elapsed[(r.frame, r.track_id)]
                                        for r in valid.itertuples()]
        # Noise floor is expressed as window displacement; not physical speed.
        valid = valid[valid.speed_px >= 1.0]
        votes = {}
        for (sec, zone, tid), g in valid.groupby(['sec', 'zone', 'track_id']):
            rad = np.radians(g.angle_deg.to_numpy(float))
            x, y = np.cos(rad).mean(), np.sin(rad).mean()
            if np.hypot(x, y) >= MIN_COHERENCE:
                votes.setdefault((int(sec), zone), []).append(
                    (float(np.degrees(np.arctan2(y, x)) % 360), float(g.speed_px_per_second.median())))
    else:
        votes=prepared_votes
    flows = {(int(r.sec), r.zone): r for r in motion.itertuples()}
    rows = []
    for obs in observations.itertuples():
        key = (int(obs.sec), obs.zone)
        people = votes.get(key, [])
        bins = dict(right=0, down=0, left=0, up=0)
        for a, _ in people:
            bins[['right', 'down', 'left', 'up'][int((a + 45) % 360 // 90)]] += 1
        n = len(people)
        coherence = direction = speed = difference = agreement = None
        if n:
            angles = np.radians([a for a, _ in people])
            x, y = np.cos(angles).mean(), np.sin(angles).mean()
            coherence = float(np.hypot(x, y))
            speed = float(np.median([s for _, s in people]))
            if coherence >= MIN_COHERENCE:
                direction = float(np.degrees(np.arctan2(y, x)) % 360)
        flow = flows.get(key)
        reason = 'insufficient_tracks'
        if n >= MIN_TRACKS:
            if direction is None:
                reason = 'mixed_individual_directions'
            elif flow is None or flow.low_confidence or pd.isna(flow.angle_deg):
                reason = 'flow_unavailable'
            elif flow.magnitude_px < 0.25:
                reason = 'weak_net_flow'
            else:
                difference = circular_difference(direction, flow.angle_deg)
                agreement = 1 - difference / 180
                reason = 'agreement' if difference <= MAX_DIFFERENCE_DEG else 'direction_disagreement'
        rows.append(dict(sec=key[0], zone=key[1], valid_track_count=n,
                         individual_angle_deg=direction if n >= MIN_TRACKS else None,
                         median_speed_px_per_second=speed, direction_coherence=coherence,
                         direction_counts=bins, circular_difference_deg=difference,
                         agreement=agreement, low_confidence=reason != 'agreement', reason=reason))
    return dict(version=HYBRID_VERSION, min_tracks=MIN_TRACKS, min_coherence=MIN_COHERENCE,
                max_difference_deg=MAX_DIFFERENCE_DEG, rows=rows,
                units={'speed': 'image pixels/second; not m/s',
                       'counts': 'distinct valid IDs per second; not entry counts'})
