"""Strict, streaming CSV validation. No inference or database imports."""
import csv
import hashlib
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

Nonnegative = Annotated[float, Field(ge=0)]
Count = Annotated[int, Field(ge=0, le=2**63-1)]
Angle = Annotated[float, Field(ge=0, lt=360)]
Unit = Annotated[float, Field(ge=0, le=1)]
RISK_MAPPING = {'낮음': 'low', '주의': 'medium', '높음': 'high'}
MAX_FILE_BYTES = 64 * 1024 * 1024
SCHEMAS = {
    'crowd_observations': ['sec', 'zone', 'count', 'inflow', 'outflow'],
    'flow': ['sec', 'zone', 'angle_deg', 'magnitude_px', 'motion', 'low_confidence'],
    'individual_directions': ['frame', 't_ms', 'track_id', 'angle_deg', 'speed_px', 'reliable'],
    'individual_zone_agg': ['sec', 'zone', 'mean_angle_deg', 'n_tracks'],
    'bottleneck_scores': ['window_start', 'zone', 'score', 'growth', 'slowdown', 'imbalance', 'risk_level'],
}


class InvalidResult(ValueError):
    pass


class Row(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

    @field_validator('low_confidence', 'reliable', mode='before', check_fields=False)
    @classmethod
    def csv_bool(cls, value):
        if type(value) is bool:
            return value
        if isinstance(value, str) and value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        raise ValueError('Expected True or False')

    @field_validator('angle_deg', 'magnitude_px', 'motion', 'speed_px', mode='before', check_fields=False)
    @classmethod
    def empty_numeric(cls, value):
        return None if value == '' else value


class ZoneSecond(Row):
    sec: Count
    zone: str = Field(min_length=1, max_length=100)


class Observation(ZoneSecond):
    count: Annotated[float, Field(ge=0, le=9999999999.9999)]
    inflow: Count
    outflow: Count

    @field_validator('count')
    @classmethod
    def storage_precision(cls, value):
        from decimal import Decimal
        if Decimal(str(value)).as_tuple().exponent < -4:
            raise ValueError('Mean count supports at most four decimal places; no silent rounding')
        return value


class Flow(ZoneSecond):
    angle_deg: Angle | None
    magnitude_px: Nonnegative | None
    motion: Nonnegative | None
    low_confidence: bool

    @model_validator(mode='after')
    def missing_flow(self):
        fields = [self.angle_deg, self.magnitude_px, self.motion]
        if any(v is None for v in fields) and not (all(v is None for v in fields) and self.low_confidence):
            raise ValueError('Missing flow requires all values missing and low_confidence=True')
        return self


class Individual(Row):
    frame: Count
    t_ms: Count
    track_id: Annotated[int, Field(gt=0, le=2**63-1)]
    angle_deg: Angle | None
    speed_px: Nonnegative | None
    reliable: bool

    @model_validator(mode='after')
    def reliability(self):
        if not self.reliable and (self.angle_deg is not None or self.speed_px is not None):
            raise ValueError('Unreliable directions must have missing angle/speed')
        if self.reliable and (self.speed_px is None or (self.angle_deg is None and self.speed_px != 0)):
            raise ValueError('Reliable missing angle is allowed only for zero displacement')
        return self


class Aggregate(ZoneSecond):
    mean_angle_deg: Angle
    n_tracks: Annotated[int, Field(gt=0, le=2**63-1)]


class Score(Row):
    window_start: Count
    zone: str = Field(min_length=1, max_length=100)
    score: Annotated[float, Field(ge=0, le=100)]
    growth: Unit
    slowdown: Unit
    imbalance: Unit
    risk_level: Literal['낮음', '주의', '높음']

class ExecutionMetadata(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    analysis_started_at: AwareDatetime
    analysis_finished_at: AwareDatetime
    model_version: str = Field(min_length=1, max_length=100)
    algorithm_version: str = Field(min_length=1, max_length=200)
    csv_schema_version: Literal['1'] = '1'
    direction_quality: Literal['unvalidated', 'validated', 'rejected'] = 'unvalidated'
    direction_review: dict = Field(default_factory=dict)
    analysis_mode: Literal['production', 'test'] = 'production'
    analysis_profile: Literal['fast', 'full'] = 'full'
    video_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=0, le=2**63-1)
    settings: dict = Field(default_factory=dict)

    @model_validator(mode='after')
    def ordered_times(self):
        if self.analysis_finished_at < self.analysis_started_at:
            raise ValueError('Analysis finish precedes start')
        if self.direction_quality == 'validated' and not all(
            isinstance(self.direction_review.get(key), str) and self.direction_review[key].strip()
            for key in ('reviewer', 'evidence')
        ):
            raise ValueError('Validated direction requires reviewer and evidence')
        return self


def validate_roi(snapshot: dict) -> tuple[str, set[str]]:
    """Validate snapshot without importing or changing the AI implementation."""
    try:
        width, height = snapshot['video_resolution']
        if type(width) is not int or type(height) is not int or min(width, height) <= 0 or max(width, height) > 2**31-1:
            raise ValueError()
        approved = snapshot['approved_for_analysis']
        status = snapshot['status']
        if type(approved) is not bool or (status, approved) not in {
            ('candidate', False), ('candidate_unvalidated', False), ('approved', True)
        }:
            raise ValueError()
        names = set()
        for zone in snapshot['zones']:
            name, polygon = zone['name'], zone['polygon']
            if not isinstance(name, str) or not name or len(name) > 100 or name in names or len(polygon) < 3:
                raise ValueError()
            names.add(name)
            for x, y in polygon:
                # Existing ROI files include vertices exactly on the image edge
                # (e.g. y=1080). These are polygon boundaries, not pixel indices.
                if type(x) not in (int, float) or type(y) not in (int, float) or not math.isfinite(x+y) or not (0 <= x <= width and 0 <= y <= height):
                    raise ValueError()
            area = sum(polygon[i][0]*polygon[(i+1)%len(polygon)][1]-polygon[(i+1)%len(polygon)][0]*polygon[i][1] for i in range(len(polygon)))
            if area == 0:
                raise ValueError()
        if not names:
            raise ValueError()
        return ('approved' if approved else 'candidate'), names
    except (KeyError, TypeError, ValueError, OverflowError):
        raise InvalidResult('Invalid ROI snapshot, candidate/approved status, or pixel bounds') from None


def validate_bundle(paths: dict[str, Path], roi: dict, metadata: ExecutionMetadata):
    if set(paths) != set(SCHEMAS):
        raise InvalidResult('Exactly five named CSV files are required')
    status, zones = validate_roi(roi)
    if roi.get('source_sha256', metadata.video_sha256) != metadata.video_sha256:
        raise InvalidResult('ROI source hash differs from execution video hash')
    for key, value in [('source_fps', metadata.fps), ('source_frame_count', metadata.frame_count)]:
        if key in roi and roi[key] != value:
            raise InvalidResult('ROI source metadata mismatch')
    models = dict(zip(SCHEMAS, [Observation, Flow, Individual, Aggregate, Score]))
    summaries, records = {}, {}
    duration = metadata.frame_count / metadata.fps
    if not math.isfinite(duration) or duration * 1000 > 2**63-1:
        raise InvalidResult('Video duration cannot be represented in milliseconds')
    for kind, path in paths.items():
        if path.stat().st_size > MAX_FILE_BYTES:
            raise InvalidResult('CSV file exceeds 64MiB')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        rows, seen, missing, count = [], set(), {name: 0 for name in SCHEMAS[kind]}, 0
        number = 1
        try:
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != SCHEMAS[kind]:
                    raise InvalidResult(f'{kind}: unexpected CSV header')
                for number, raw in enumerate(reader, 2):
                    row = models[kind].model_validate(raw)
                    item = row.model_dump()
                    if kind == 'individual_directions':
                        if row.frame >= metadata.frame_count or row.t_ms != int(row.frame / metadata.fps * 1000):
                            raise ValueError('Invalid frame or millisecond clock')
                        key = (row.frame, row.track_id)
                    else:
                        second = getattr(row, 'sec', getattr(row, 'window_start', None))
                        if row.zone not in zones or second >= duration:
                            raise ValueError('Unknown zone or offset outside video')
                        key = (second, row.zone)
                    if key in seen:
                        raise ValueError('Duplicate CSV key')
                    seen.add(key)
                    for name, value in item.items():
                        missing[name] += value is None
                    count += 1
                    if count > (1_000_000 if kind == 'individual_directions' else 100_000):
                        raise ValueError('CSV row limit exceeded')
                    if kind != 'individual_directions':
                        rows.append(item)
        except (ValidationError, ValueError, UnicodeError, csv.Error) as exc:
            # Do not reflect raw uploaded cells or sensitive file paths.
            raise InvalidResult(f'{kind}: invalid data at CSV row {locals().get("number", 1)} ({type(exc).__name__})') from None
        summaries[kind] = {'sha256': digest, 'bytes': path.stat().st_size, 'rows': count, 'missing': missing}
        records[kind] = rows
    # Empty score/raw direction/aggregate files are legitimate. Observations must
    # cover every configured zone at least once, even for an empty scene.
    if set(r['zone'] for r in records['crowd_observations']) != zones:
        raise InvalidResult('Observations must cover all configured zones')
    if metadata.analysis_profile == 'fast' and any(summaries[k]['rows'] for k in ('flow','bottleneck_scores')):
        raise InvalidResult('FAST results must not contain Optical Flow or bottleneck score rows')
    return status, summaries, records
