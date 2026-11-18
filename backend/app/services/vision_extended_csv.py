"""Lightweight validation for the supplementary Vision AI result files.

Deliberately not as exhaustive as vision_csv.py (no duplicate-key/zone-membership/row-limit
checks) — these files are additive extensions beyond the original FR-ZON spec and are never
required for an analysis job to succeed (see worker.py, vision_extended_results.py). The goal
here is just to catch a malformed or truncated file before it goes into vision_extended_results,
not to match the core five-file contract's rigor.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Nonnegative = Annotated[float, Field(ge=0)]
Unit = Annotated[float, Field(ge=0, le=1)]
Angle = Annotated[float, Field(ge=0, lt=360)]
MAX_FILE_BYTES = 64 * 1024 * 1024

CSV_SCHEMAS = {
    'dwell_times': ['track_id', 'zone', 'enter_sec', 'exit_sec', 'dwell_s'],
    'counter_flow': ['sec', 'zone', 'mean_angle_deg', 'reverse_ratio', 'n_tracks'],
    'queue_scores': ['window_start', 'zone', 'queue_score', 'density', 'stillness', 'is_queue'],
    'density_heatmap': ['window_start', 'cell_x', 'cell_y', 'count'],
    'crowd_speed_zone': ['sec', 'zone', 'speed_mps', 'low_confidence'],
    'crowd_speed_individual': ['frame', 't_ms', 'track_id', 'speed_mps', 'reliable'],
}
REQUIRED_KINDS = {'dwell_times', 'counter_flow', 'queue_scores', 'density_heatmap'}
OPTIONAL_KINDS = {'crowd_speed_zone', 'crowd_speed_individual'}


class InvalidExtendedResult(ValueError):
    pass


class _Row(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

    @field_validator('low_confidence', 'reliable', 'is_queue', mode='before', check_fields=False)
    @classmethod
    def csv_bool(cls, value):
        if type(value) is bool:
            return value
        if isinstance(value, str) and value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        raise ValueError('Expected True or False')

    @field_validator('mean_angle_deg', 'reverse_ratio', 'speed_mps', mode='before', check_fields=False)
    @classmethod
    def empty_numeric(cls, value):
        return None if value == '' else value


class DwellTime(_Row):
    track_id: Annotated[int, Field(gt=0, le=2**63 - 1)]
    zone: str = Field(min_length=1, max_length=100)
    enter_sec: Annotated[int, Field(ge=0)]
    exit_sec: Annotated[int, Field(ge=0)]
    dwell_s: Nonnegative


class CounterFlow(_Row):
    sec: Annotated[int, Field(ge=0)]
    zone: str = Field(min_length=1, max_length=100)
    mean_angle_deg: Angle
    reverse_ratio: Unit | None
    n_tracks: Annotated[int, Field(ge=0)]


class QueueScore(_Row):
    window_start: Annotated[int, Field(ge=0)]
    zone: str = Field(min_length=1, max_length=100)
    queue_score: Annotated[float, Field(ge=0, le=100)]
    density: Unit
    stillness: Unit
    is_queue: bool


class DensityHeatmapCell(_Row):
    window_start: Annotated[int, Field(ge=0)]
    cell_x: Annotated[int, Field(ge=0)]
    cell_y: Annotated[int, Field(ge=0)]
    count: Nonnegative


class CrowdSpeedZone(_Row):
    sec: Annotated[int, Field(ge=0)]
    zone: str = Field(min_length=1, max_length=100)
    speed_mps: Nonnegative | None
    low_confidence: bool


class CrowdSpeedIndividual(_Row):
    frame: Annotated[int, Field(ge=0)]
    t_ms: Annotated[int, Field(ge=0)]
    track_id: Annotated[int, Field(gt=0, le=2**63 - 1)]
    speed_mps: Nonnegative | None
    reliable: bool


CSV_MODELS = {
    'dwell_times': DwellTime,
    'counter_flow': CounterFlow,
    'queue_scores': QueueScore,
    'density_heatmap': DensityHeatmapCell,
    'crowd_speed_zone': CrowdSpeedZone,
    'crowd_speed_individual': CrowdSpeedIndividual,
}


class TrackingQuality(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    track_count: Annotated[int, Field(ge=0)]
    mean_track_length_frames: Nonnegative
    median_track_length_frames: Nonnegative
    mean_confidence: Annotated[float, Field(ge=0, le=1)]
    min_confidence: Annotated[float, Field(ge=0, le=1)]
    suspected_id_switch_ratio: Unit


def _read_csv_rows(path: Path, kind: str) -> list[dict]:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise InvalidExtendedResult(f'{kind}: file exceeds 64MiB')
    model = CSV_MODELS[kind]
    rows = []
    try:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != CSV_SCHEMAS[kind]:
                raise InvalidExtendedResult(f'{kind}: unexpected CSV header')
            for raw in reader:
                rows.append(model.model_validate(raw).model_dump())
    except (ValidationError, ValueError, UnicodeError, csv.Error) as exc:
        raise InvalidExtendedResult(f'{kind}: invalid data ({type(exc).__name__})') from None
    return rows


def validate_extended_bundle(paths: dict[str, Path]) -> tuple[dict[str, dict], dict[str, object]]:
    """paths: dwell_times/counter_flow/queue_scores/density_heatmap(csv) + tracking_quality(json),
    optionally crowd_speed_zone/crowd_speed_individual(csv).

    반환: (manifest: {kind: {sha256, bytes, rows}}, records: {kind: parsed rows/dict}).
    """
    present = set(paths)
    if not REQUIRED_KINDS <= present or present - REQUIRED_KINDS - OPTIONAL_KINDS - {'tracking_quality'}:
        raise InvalidExtendedResult('Missing required files or unexpected extra files')
    if 'tracking_quality' not in present:
        raise InvalidExtendedResult('tracking_quality.json is required')

    manifest, records = {}, {}
    for kind, path in paths.items():
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if kind == 'tracking_quality':
            try:
                data = TrackingQuality.model_validate_json(path.read_text(encoding='utf-8')).model_dump()
            except ValidationError as exc:
                raise InvalidExtendedResult(f'tracking_quality: invalid data ({type(exc).__name__})') from None
            records[kind] = data
            manifest[kind] = {'sha256': digest, 'bytes': path.stat().st_size, 'rows': 1}
        else:
            rows = _read_csv_rows(path, kind)
            records[kind] = rows
            manifest[kind] = {'sha256': digest, 'bytes': path.stat().st_size, 'rows': len(rows)}
    return manifest, records
