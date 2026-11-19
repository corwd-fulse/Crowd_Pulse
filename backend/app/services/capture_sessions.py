"""Metadata validation and causal, camera-long training rows. No model fitting."""
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, model_validator

INTERVALS = (1, 5, 10, 30, 60)


class Clip(BaseModel):
    model_config = ConfigDict(extra='forbid')
    camera_id: str = Field(min_length=1, max_length=80)
    video_id: UUID
    job_id: UUID | None = None
    roi_name: str = Field(default='', max_length=100)
    zone_id: str | None = None
    target_zone_id: str | None = None
    capture_start: datetime | None = None
    clock_offset_ms: int = Field(default=0, ge=-86400000, le=86400000)
    clock_verified: bool = False
    clock_evidence: str = Field(default='', max_length=1000)
    active: bool = True

    @model_validator(mode='after')
    def clock(self):
        if self.capture_start and self.capture_start.utcoffset() is None:
            raise ValueError('Capture time requires timezone')
        if self.clock_verified and (not self.capture_start or not self.clock_evidence.strip()):
            raise ValueError('Verified clock requires capture time and evidence')
        if self.job_id and not self.roi_name.strip():
            raise ValueError('Analysis binding requires an immutable ROI name')
        return self


class CaptureData(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1, max_length=100)
    start: datetime
    end: datetime
    timing_basis: Literal['planned', 'recorded'] = 'planned'
    mode: Literal['independent', 'scenario', 'synchronized'] = 'independent'
    is_holiday: bool | None = None
    holiday_evidence: str = Field(default='', max_length=500)
    is_event: bool = False
    event_name: str = Field(default='', max_length=100)
    event_phase: Literal['normal', 'before_event', 'opening', 'during_event', 'closing', 'after_event'] = 'normal'
    lighting_condition: Literal['day', 'night', 'mixed', 'unknown'] = 'unknown'
    environment: str = Field(default='', max_length=4000)
    clips: list[Clip] = Field(default_factory=list, max_length=256)

    @model_validator(mode='after')
    def validate_session(self):
        if self.start.utcoffset() is None or self.end.utcoffset() is None or self.end <= self.start:
            raise ValueError('Ordered, timezone-aware start/end required')
        if (self.end - self.start).total_seconds() > 7 * 86400:
            raise ValueError('A capture session is limited to seven days')
        if not self.is_event and (self.event_phase != 'normal' or self.event_name):
            raise ValueError('A non-event session uses normal phase without event name')
        if self.is_holiday is not None and not self.holiday_evidence.strip():
            raise ValueError('Holiday classification requires a calendar source')
        active = [c for c in self.clips if c.active]
        # Conservative: multiple recordings from the same camera require separate sessions.
        if len({c.camera_id for c in active}) != len(active):
            raise ValueError('Only one active clip per camera; split sequential recordings into sessions')
        if len({c.video_id for c in active}) != len(active):
            raise ValueError('A video cannot represent multiple independent cameras')
        if self.mode == 'synchronized' and (self.timing_basis != 'recorded' or any(not c.clock_verified for c in active)):
            raise ValueError('Synchronized observations require recorded times and verified camera clocks')
        return self


COLUMNS = ['timestamp', 'video_offset_ms', 'session_id', 'session_revision', 'venue_id', 'venue_revision',
    'camera_id', 'video_id', 'job_id', 'result_fingerprint', 'zone_id', 'roi_name', 'target_zone_id',
    'person_count', 'inflow', 'outflow', 'net_flow', 'direction_count', 'reverse_direction_count',
    'motion', 'bottleneck_score', 'count_change', 'rolling_mean', 'is_weekend', 'is_holiday',
    'is_event', 'event_phase', 'lighting_condition', 'low_confidence', 'observed_seconds', 'interval_seconds']


def training_rows(metrics, clip, session, identity, interval, *, ordered=False):
    """Only complete past buckets are features; nulls are never converted to zero.

    Input offset denotes the START of an existing one-second metric. Output time
    denotes bucket END. Duplicate or missing seconds invalidate that bucket.
    No cross-camera totals: overlapping coverage needs independent evidence.
    """
    if interval not in INTERVALS:
        raise ValueError('Unsupported interval')
    from itertools import groupby, islice
    if not ordered:metrics=sorted(metrics,key=lambda m:m['video_offset_ms'])
    groups=iter(groupby(metrics,key=lambda m:m['video_offset_ms']//(interval*1000)))
    history=deque(maxlen=5);previous=None;last=None;emitted=0
    def buckets():
        nonlocal last,emitted
        for bucket,group in groups:
            if last is not None and bucket<=last:raise ValueError('Training metrics must be ordered')
            if last is not None and bucket-last>250000:raise ValueError('Sparse export exceeds the row budget')
            for missing in range(last+1 if last is not None else bucket,bucket):
                emitted+=1
                if emitted>250000:raise ValueError('Export exceeds row budget')
                yield missing,[]
            values=list(islice(group,interval+2))
            if len(values)>interval+1:raise ValueError('Too many duplicate observations in one bucket')
            emitted+=1
            if emitted>250000:raise ValueError('Export exceeds row budget')
            yield bucket,values
            last=bucket
    for bucket,values in buckets():
        offsets = [v['video_offset_ms'] for v in values]
        expected = {bucket * interval * 1000 + n * 1000 for n in range(interval)}
        complete = len(offsets) == interval and set(offsets) == expected and all(v.get('complete_second', True) for v in values)
        def aggregate(key, mean=False):
            numbers = [v.get(key) for v in values]
            if not complete or any(n is None for n in numbers):
                return None
            if key in ('inflow', 'outflow'):
                return sum(int(n) for n in numbers)
            return sum(float(n) for n in numbers) / (interval if mean else 1)
        count = aggregate('mean_person_count', True)
        incoming, outgoing = aggregate('inflow'), aggregate('outflow')
        confidence = [v.get('low_confidence') for v in values]
        low = True if True in confidence else (False if complete and all(v is False for v in confidence) else None)
        if previous is None or bucket != previous[0] + 1 or count is None:
            history.clear()
        change = count - previous[1] if count is not None and previous and previous[1] is not None and bucket == previous[0] + 1 else None
        if count is not None:
            history.append(count)
        offset_end = (bucket + 1) * interval * 1000
        actual = session.timing_basis == 'recorded' and clip.clock_verified and clip.capture_start is not None
        timestamp = clip.capture_start + timedelta(milliseconds=clip.clock_offset_ms + offset_end) if actual else None
        row = dict.fromkeys(COLUMNS)
        local_time = timestamp.astimezone(session.start.tzinfo) if timestamp else None
        row.update(identity, camera_id=clip.camera_id, video_id=str(clip.video_id), job_id=str(clip.job_id) if clip.job_id else None,
            zone_id=clip.zone_id, roi_name=clip.roi_name, target_zone_id=clip.target_zone_id,
            timestamp=timestamp.isoformat() if timestamp else None, video_offset_ms=offset_end,
            person_count=count, inflow=incoming, outflow=outgoing,
            net_flow=incoming-outgoing if incoming is not None and outgoing is not None else None,
            motion=aggregate('motion', True) if low is False else None,
            count_change=change, rolling_mean=sum(history)/len(history) if history else None,
            is_weekend=local_time.weekday() >= 5 if local_time else None,
            is_holiday=session.is_holiday if local_time and local_time.date() == session.start.date() else None,
            is_event=session.is_event, event_phase=session.event_phase, lighting_condition=session.lighting_condition,
            low_confidence=low, observed_seconds=len({v['video_offset_ms'] for v in values
                if v.get('mean_person_count') is not None and v.get('complete_second', True)}), interval_seconds=interval)
        previous = (bucket, count)
        yield row


def dataset_schema():
    return {'version': 'capture-long-v1', 'columns': COLUMNS, 'intervals_seconds': INTERVALS,
        'row_unit': 'camera / immutable ROI / completed interval; never sum overlapping cameras',
        'timestamp': 'verified absolute bucket END, otherwise null; relative offset remains available',
        'missing': 'CSV empty cell / JSON null; no zero imputation',
        'rolling_mean': 'up to five contiguous completed buckets, trailing only',
        'unavailable_features': {'direction_count': 'requires target-specific calibrated gate events',
            'reverse_direction_count': 'requires target-specific calibrated gate events',
            'bottleneck_score': 'legacy whole-video normalization is excluded to prevent future leakage'},
        'target_labels': 'not generated; this export is feature preparation, not a trained forecast model',
        'types': {'timestamp': 'ISO8601 with timezone or null', 'video_offset_ms': 'integer bucket end',
            'person_count': 'float mean over complete bucket or null', 'inflow': 'integer sum or null',
            'outflow': 'integer sum or null', 'net_flow': 'integer or null', 'motion': 'float or null',
            'low_confidence': 'boolean or null', 'is_weekend': 'boolean or null', 'is_holiday': 'boolean or null',
            'is_event': 'boolean', 'identifiers': 'strings; UUID for session/video/job'},
        'split_rule': 'split by session/date before preprocessing; keep simultaneous cameras in the same split'}
