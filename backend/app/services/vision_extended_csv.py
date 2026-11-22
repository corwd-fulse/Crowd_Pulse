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

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.services.vision_csv import MAX_FILE_BYTES, Row as _CoreRow

Nonnegative = Annotated[float, Field(ge=0)]
Unit = Annotated[float, Field(ge=0, le=1)]
Angle = Annotated[float, Field(ge=0, lt=360)]

CSV_SCHEMAS = {
    'dwell_times': ['track_id', 'zone', 'enter_sec', 'exit_sec', 'dwell_s'],
    'counter_flow': ['sec', 'zone', 'mean_angle_deg', 'reverse_ratio', 'n_tracks'],
    'queue_scores': ['window_start', 'zone', 'queue_score', 'density', 'stillness', 'is_queue'],
    'density_heatmap': ['window_start', 'cell_x', 'cell_y', 'count'],
    'crowd_speed_zone': ['sec', 'zone', 'speed_mps', 'low_confidence'],
    'crowd_speed_individual': ['frame', 't_ms', 'track_id', 'speed_mps', 'reliable'],
    'dmcount_estimates': ['window_start', 'zone', 'estimated_count', 'inference_ms', 'model_version', 'density_map_path'],
}
REQUIRED_KINDS = {'dwell_times', 'counter_flow', 'queue_scores', 'density_heatmap'}
OPTIONAL_KINDS = {'crowd_speed_zone', 'crowd_speed_individual', 'dmcount_estimates'}


class InvalidExtendedResult(ValueError):
    pass


class _Row(_CoreRow):
    """vision_csv.Row의 csv_bool/empty_numeric을 그대로 물려받는다(bool/빈문자열 파싱 규칙이
    두 군데서 따로 어긋나지 않도록). 여기서는 core에 없는 필드 이름(is_queue 등)에 대해서만
    같은 검증을 추가로 붙인다."""

    @field_validator('is_queue', mode='before', check_fields=False)
    @classmethod
    def csv_bool_extended(cls, value):
        return cls.csv_bool(value)

    @field_validator('mean_angle_deg', 'reverse_ratio', 'speed_mps', mode='before', check_fields=False)
    @classmethod
    def empty_numeric_extended(cls, value):
        return cls.empty_numeric(value)


class DwellTime(_Row):
    track_id: Annotated[int, Field(gt=0, le=2**63 - 1)]
    zone: str = Field(min_length=1, max_length=100)
    enter_sec: Annotated[int, Field(ge=0)]
    exit_sec: Annotated[int, Field(ge=0)]
    dwell_s: Nonnegative

    @model_validator(mode='after')
    def ordered_times(self):
        # dwell_s는 프레임 단위 정밀 계산이고 enter_sec/exit_sec는 그 초 단위로 내림한 값이라
        # dwell_s == exit_sec - enter_sec가 정확히 맞아야 하는 건 아니다(둘 다 같은 체류를
        # 다른 정밀도로 표현한 값). 그래도 순서가 뒤집히는 것(exit < enter)은 항상 데이터 오류다.
        if self.exit_sec < self.enter_sec:
            raise ValueError('exit_sec precedes enter_sec')
        return self


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


class DMCountEstimate(_Row):
    window_start: Annotated[int, Field(ge=0)]
    zone: str = Field(min_length=1, max_length=100)
    estimated_count: Nonnegative
    inference_ms: Nonnegative
    model_version: str = Field(min_length=1, max_length=100)
    # density_estimation.py가 heatmap_dir 없이 돌면 전부 빈 문자열(이미지 저장 안 함).
    density_map_path: Annotated[str, Field(max_length=500)] | None

    @field_validator('density_map_path', mode='before')
    @classmethod
    def empty_path_is_none(cls, value):
        return None if value == '' else value


CSV_MODELS = {
    'dwell_times': DwellTime,
    'counter_flow': CounterFlow,
    'queue_scores': QueueScore,
    'density_heatmap': DensityHeatmapCell,
    'crowd_speed_zone': CrowdSpeedZone,
    'crowd_speed_individual': CrowdSpeedIndividual,
    'dmcount_estimates': DMCountEstimate,
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
