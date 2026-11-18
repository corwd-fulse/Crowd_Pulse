"""Versioned manual camera/gate semantics, separate from video ROI pixels."""
import math
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, model_validator

Number = Annotated[float, Field(allow_inf_nan=False)]
MapNumber = Annotated[float, Field(ge=0, le=1000, allow_inf_nan=False)]
Pixel = Annotated[float, Field(ge=0, le=20000, allow_inf_nan=False)]


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class VenueCamera(Strict):
    camera_id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=100)
    registered_camera_id: int | None = Field(default=None, gt=0)
    point: tuple[MapNumber, MapNumber]
    position_confirmed: bool = False
    bearing_deg: Number | None = Field(default=None, ge=0, lt=360)
    active: bool = True
    zone_id: str = ''
    job_id: UUID | None = None
    roi_name: str = Field(default='', max_length=100)
    coverage: list[tuple[MapNumber, MapNumber]] = Field(default_factory=list, max_length=100)
    full_zone_coverage: bool = False
    capture_start: datetime | None = None
    clock_verified: bool = False
    clock_evidence: str = Field(default='', max_length=1000)
    scenario_offset_s: Number = Field(default=0, ge=0)

    @model_validator(mode='after')
    def clock_and_coverage(self):
        if self.job_id is None and (self.roi_name or self.full_zone_coverage or self.clock_verified):
            raise ValueError('Unbound camera cannot assert ROI coverage or a verified video clock')
        if self.capture_start and self.capture_start.tzinfo is None:
            raise ValueError('Capture start must include timezone')
        if self.clock_verified and (not self.capture_start or not self.clock_evidence.strip()):
            raise ValueError('Verified clock requires capture time and evidence')
        if self.coverage:
            if len(self.coverage) < 3:
                raise ValueError('Coverage polygon needs at least three points')
            from app.services.roi_geometry import validate_disjoint
            validate_disjoint([{'polygon': self.coverage}])
        return self


class Corridor(Strict):
    corridor_id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=100)
    camera_id: str
    from_zone_id: str
    to_zone_id: str
    # Directed finite line in original video pixel space, plus a point on the center side.
    boundary: tuple[tuple[Pixel, Pixel], tuple[Pixel, Pixel]]
    center_side: tuple[Pixel, Pixel]
    role: Literal['approach', 'center_boundary'] = 'approach'
    boundary_evidence: str = Field(default='', max_length=1000)
    boundary_verified: bool = False
    counting_group: str = Field(min_length=1, max_length=80)
    independent_counting_verified: bool = False
    independence_evidence: str = Field(default='', max_length=1000)
    arrival_ratio: Number | None = Field(default=None, ge=0, le=1)
    travel_seconds: Number | None = Field(default=None, ge=0, le=3600)
    arrival_basis: Literal['unknown', 'assumption', 'measured'] = 'unknown'
    arrival_evidence: str = Field(default='', max_length=1000)

    @model_validator(mode='after')
    def gate_geometry(self):
        a,b=self.boundary
        cross=(b[0]-a[0])*(self.center_side[1]-a[1])-(b[1]-a[1])*(self.center_side[0]-a[0])
        if math.dist(a,b)<1 or abs(cross)<1e-6:
            raise ValueError('Nonzero boundary and off-line center-side point required')
        if self.independent_counting_verified and not self.independence_evidence.strip():
            raise ValueError('Independent counting needs an explanation')
        if self.role=='center_boundary' and not self.boundary_evidence.strip():
            raise ValueError('A physical center boundary needs explicit evidence')
        if self.boundary_verified and not self.boundary_evidence.strip():
            raise ValueError('A verified boundary needs explicit evidence')
        if self.arrival_basis!='unknown' and (self.arrival_ratio is None or self.travel_seconds is None or not self.arrival_evidence.strip()):
            raise ValueError('Arrival assumptions/measurements need ratio, travel time and evidence')
        return self


class ConcentrationPolicy(Strict):
    window_seconds: int = Field(default=30, ge=10, le=120)
    center_observer_camera_id: str | None = None
    center_boundary_coverage_complete: bool = False
    coverage_evidence: str = Field(default='', max_length=1000)
    # Optional administrator proposal: no automatic safety thresholds.
    attention_net_per_minute: Number | None = Field(default=None, gt=0)
    caution_capacity_ratio: Number | None = Field(default=None, gt=0, le=2)
    urgent_capacity_ratio: Number | None = Field(default=None, gt=0, le=3)
    thresholds_validated: bool = False
    threshold_evidence: str = Field(default='', max_length=1000)

    @model_validator(mode='after')
    def evidence(self):
        if self.center_boundary_coverage_complete and not self.coverage_evidence.strip():
            raise ValueError('Complete center boundary coverage requires evidence')
        if self.thresholds_validated and not self.threshold_evidence.strip():
            raise ValueError('Validated thresholds require evidence')
        if self.caution_capacity_ratio and self.urgent_capacity_ratio and self.urgent_capacity_ratio<=self.caution_capacity_ratio:
            raise ValueError('Urgent ratio must exceed caution ratio')
        return self


class EstimateRequest(Strict):
    revision: int = Field(gt=0)
    target_zone_id: str | None = Field(default=None, max_length=80)
    at: datetime | None = None
    offset_seconds: Number = Field(default=0, ge=0)

    @model_validator(mode='after')
    def timezone(self):
        if self.at and self.at.tzinfo is None:
            raise ValueError('Observation time requires timezone')
        return self
