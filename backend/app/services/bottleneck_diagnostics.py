"""Explain score availability without changing scores, thresholds or saved data.

Standard library only: safe for the read-only API as well as the AI pipeline.
Hybrid confidence is deliberately reported separately from score prerequisites.
"""
from collections import Counter, defaultdict
import math

REQUIRED_METRICS = ('count', 'motion', 'inflow', 'outflow')


def _missing(value):
    return value is None or not math.isfinite(float(value))


def describe_bottleneck_windows(records, score_keys, *, analysis_profile='full', window_s=10):
    """Records use sec/zone/count/motion/inflow/outflow and optional hybrid/low_confidence.

    Availability always follows the actual saved/computed score keys. This helper
    does not invent a score or retroactively apply the new policy to old scores.
    """
    groups = defaultdict(dict)
    for row in records:
        start = int(row['sec']) // window_s * window_s
        groups[row['zone']].setdefault(start, []).append(row)
    result = []
    for zone, windows in sorted(groups.items()):
        previous = None
        for start, rows in sorted(windows.items()):
            missing = {key: sorted(int(r['sec']) for r in rows if _missing(r.get(key)))
                       for key in REQUIRED_METRICS}
            low = sorted(int(r['sec']) for r in rows if r.get('low_confidence') == True)
            reasons = []
            if analysis_profile == 'fast':
                reasons.append('fast_not_run')
            else:
                if len(windows) < 2:
                    reasons.append('insufficient_windows')
                if low:
                    reasons.append('low_confidence_flow')
                reasons.extend('missing_' + key for key, seconds in missing.items() if seconds)
                if previous and any(previous['missing_metrics'].values()):
                    reasons.append('previous_window_missing_metrics')
            people = [r['hybrid'] for r in rows if isinstance(r.get('hybrid'), dict)]
            coherent = [r['direction_coherence'] for r in people if r.get('direction_coherence') is not None]
            analyzed = (start, zone) in score_keys
            result.append(dict(
                window_start=start, zone=zone, status='analyzed' if analyzed else 'unanalyzed',
                reason=None if analyzed else (reasons[0] if reasons else 'score_unavailable'),
                reasons=[] if analyzed else reasons or ['score_unavailable'],
                missing_metrics=missing, flow_low_confidence_seconds=low,
                observed_seconds=sorted(int(r['sec']) for r in rows),
                previous_window_start=previous['window_start'] if previous else None,
                previous_missing_metrics=previous['missing_metrics'] if previous else {},
                hybrid_affects_bottleneck=False,
                hybrid=dict(reason_counts=dict(Counter(r['reason'] for r in people)),
                            valid_tracks_min=min((r['valid_track_count'] for r in people), default=None),
                            valid_tracks_max=max((r['valid_track_count'] for r in people), default=None),
                            coherence_min=min(coherent, default=None), coherence_max=max(coherent, default=None),
                            issue_seconds={reason: sorted(int(r['sec']) for r in rows
                                if isinstance(r.get('hybrid'), dict) and r['hybrid'].get('reason') == reason)
                                for reason in ('insufficient_tracks', 'weak_net_flow', 'direction_disagreement',
                                               'mixed_individual_directions', 'flow_unavailable')})))
            previous = result[-1]
    return result
