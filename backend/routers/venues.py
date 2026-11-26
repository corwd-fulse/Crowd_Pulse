"""Manual semantic mapping only: never project image pixels into floorplan space."""
import base64
from uuid import UUID
from typing import Annotated, Literal
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ConfigDict, model_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.venue import VenueRevision
from app.models.vision import VisionResult
from app.services.roi_geometry import validate_disjoint
from app.services.concentration_schema import VenueCamera, Corridor, ConcentrationPolicy

router = APIRouter(prefix='/api/venues', tags=['Venue control tower'])
Coordinate = Annotated[float, Field(ge=0, le=1000, allow_inf_nan=False)]


class Mapping(BaseModel):
    model_config = ConfigDict(extra='forbid')
    job_id: UUID
    roi_name: str = Field(min_length=1, max_length=100)


class Zone(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=100)
    polygon: list[tuple[Coordinate, Coordinate]] = Field(min_length=3, max_length=100)
    color: str = Field(pattern=r'^#[0-9a-fA-F]{6}$')
    visible: bool = True
    capacity: int | None = Field(default=None, gt=0, le=1000000)
    mapping: Mapping | None = None
    note: str = Field(default='', max_length=4000)


class Marker(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=100)
    kind: Literal['entrance', 'facility', 'camera']
    point: tuple[Coordinate, Coordinate]


class VenueLink(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=100)
    from_zone_id: str
    to_zone_id: str
    kind: Literal['walkway', 'entrance', 'stairs', 'bottleneck'] = 'walkway'
    bidirectional: bool = True
    camera_ids: list[str] = Field(default_factory=list, max_length=32)


class Configuration(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1, max_length=100)
    coordinate_system: Literal['venue_normalized_1000'] = 'venue_normalized_1000'
    venue_type: str = Field(default='unspecified', min_length=1, max_length=80)
    image: str | None = Field(default=None, max_length=7000000)
    width: int = Field(default=1600, ge=1, le=20000)
    height: int = Field(default=1000, ge=1, le=20000)
    zones: list[Zone] = Field(default_factory=list, max_length=100)
    markers: list[Marker] = Field(default_factory=list, max_length=100)
    address: str = Field(default='', max_length=500)
    center_zone_id: str | None = None
    integration_mode: Literal['independent', 'scenario', 'synchronized'] = 'independent'
    cameras: list[VenueCamera] = Field(default_factory=list, max_length=32)
    corridors: list[Corridor] = Field(default_factory=list, max_length=64)
    concentration_policy: ConcentrationPolicy = Field(default_factory=ConcentrationPolicy)
    target_policies: dict[str, ConcentrationPolicy] = Field(default_factory=dict, max_length=100)
    links: list[VenueLink] = Field(default_factory=list, max_length=512)

    @model_validator(mode='after')
    def geometry_and_image(self):
        if len({z.id for z in self.zones}) != len(self.zones) or len({z.name.strip() for z in self.zones}) != len(self.zones):
            raise ValueError('Duplicate zone ID/name')
        validate_disjoint([z.model_dump() for z in self.zones])
        zone_ids={z.id for z in self.zones}
        camera_ids={c.camera_id for c in self.cameras}
        if len({link.id for link in self.links}) != len(self.links):
            raise ValueError('Duplicate venue link ID')
        if any(link.from_zone_id not in zone_ids or link.to_zone_id not in zone_ids or link.from_zone_id == link.to_zone_id
               or not set(link.camera_ids).issubset(camera_ids) for link in self.links):
            raise ValueError('Venue links require distinct existing zones and registered venue cameras')
        if len(camera_ids)!=len(self.cameras) or len({g.corridor_id for g in self.corridors})!=len(self.corridors):
            raise ValueError('Duplicate camera/corridor ID')
        if self.center_zone_id is not None and self.center_zone_id not in zone_ids:
            raise ValueError('Unknown center zone')
        if any(c.zone_id and c.zone_id not in zone_ids for c in self.cameras):
            raise ValueError('Unknown camera zone')
        targets = set(self.target_policies) | ({self.center_zone_id} if self.center_zone_id else set())
        if not set(self.target_policies).issubset(zone_ids):
            raise ValueError('Unknown prediction target zone')
        if any(g.camera_id not in camera_ids or g.from_zone_id not in zone_ids or g.to_zone_id not in targets or g.from_zone_id==g.to_zone_id for g in self.corridors):
            raise ValueError('Corridors must connect an existing origin to the selected center')
        for target, policy in self.target_policies.items():
            if policy.center_observer_camera_id and not any(c.camera_id == policy.center_observer_camera_id and c.zone_id == target and c.active for c in self.cameras):
                raise ValueError('Target observer must be an active camera in its target zone')
        for gate in self.corridors:
            camera = next(c for c in self.cameras if c.camera_id == gate.camera_id)
            if not camera.job_id:
                raise ValueError('Gate calibration requires a bound video analysis')
        observer=self.concentration_policy.center_observer_camera_id
        if observer and not any(c.camera_id==observer and c.zone_id==self.center_zone_id for c in self.cameras):
            raise ValueError('Center observer must cover the center zone')
        pairs = [(str(z.mapping.job_id), z.mapping.roi_name) for z in self.zones if z.mapping]
        if len(set(pairs)) != len(pairs):
            raise ValueError('One observed ROI can map to only one venue zone')
        if self.image:
            prefix, _, encoded = self.image.partition(',')
            if prefix not in ('data:image/png;base64', 'data:image/jpeg;base64'):
                raise ValueError('Only raster PNG/JPEG storage supported; rasterize SVG before upload')
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) > 5 * 1024 * 1024 or not (raw.startswith(b'\x89PNG\r\n\x1a\n') or raw.startswith(b'\xff\xd8\xff')):
                raise ValueError('Invalid or oversized floorplan')
        return self


class SaveRevision(BaseModel):
    base_revision: int = Field(ge=0)
    configuration: Configuration


def latest(db, venue_id):
    return db.scalar(select(VenueRevision).where(VenueRevision.venue_id == venue_id).order_by(VenueRevision.revision.desc()).limit(1))


def view(row):
    return {'venue_id': row.venue_id, 'revision': row.revision, 'configuration': row.configuration, 'created_at': row.created_at}


@router.get('/{venue_id}')
def get_venue(venue_id: str, db: Session = Depends(get_db)):
    row = latest(db, venue_id)
    if row is None:
        raise HTTPException(404, 'Venue not configured')
    return view(row)


@router.get('/{venue_id}/revisions')
def history(venue_id: str, db: Session = Depends(get_db)):
    rows = db.scalars(select(VenueRevision).where(VenueRevision.venue_id == venue_id).order_by(VenueRevision.revision.desc()).limit(100)).all()
    return [{'revision': r.revision, 'created_at': r.created_at} for r in rows]


@router.get('/{venue_id}/revisions/{revision}')
def get_revision(venue_id: str, revision: int, db: Session = Depends(get_db)):
    row = db.get(VenueRevision, (venue_id, revision))
    if row is None:
        raise HTTPException(404, 'Revision not found')
    return view(row)


@router.post('/{venue_id}/revisions', status_code=201)
def save(venue_id: str, payload: SaveRevision, db: Session = Depends(get_db)):
    if not venue_id or len(venue_id) > 80:
        raise HTTPException(422, 'Invalid venue ID')
    current = latest(db, venue_id)
    if payload.base_revision != (current.revision if current else 0):
        raise HTTPException(409, 'Configuration changed; reload before saving')
    configuration = payload.configuration.model_dump(mode='json')
    from app.models import AnalysisJob, Video, Camera, CaptureVideoVenue
    existing_jobs = set()
    if current:
        existing_jobs = {c['job_id'] for c in current.configuration.get('cameras', []) if c.get('job_id')}
        existing_jobs |= {z['mapping']['job_id'] for z in current.configuration.get('zones', []) if z.get('mapping')}

    def check_new_source(job_id):
        # Preserve historical mappings; new assignments respect capture ownership.
        if str(job_id) in existing_jobs:
            return
        job = db.get(AnalysisJob, job_id)
        owner = db.get(CaptureVideoVenue, job.video_id) if job else None
        if owner and owner.venue_id != venue_id:
            raise HTTPException(409, 'Source video belongs to another venue')

    snapshots = {}
    for zone in payload.configuration.zones:
        if not zone.mapping:
            continue
        check_new_source(zone.mapping.job_id)
        result = db.get(VisionResult, zone.mapping.job_id)
        if result is None or zone.mapping.roi_name not in {z['name'] for z in result.roi_snapshot['zones']}:
            raise HTTPException(422, 'Mapping must reference a stored analysis ROI')
        # Preserve the exact ROI used by this mapping revision. Never change the job.
        snapshots[str(zone.mapping.job_id)] = result.roi_snapshot
    camera_sources={}
    for camera in payload.configuration.cameras:
        if camera.registered_camera_id and db.get(Camera,camera.registered_camera_id) is None:
            raise HTTPException(422,'Registered camera not found')
        if camera.job_id is None:
            continue
        check_new_source(camera.job_id)
        job=db.get(AnalysisJob,camera.job_id)
        result=db.get(VisionResult,camera.job_id)
        if job is None or result is None or job.status!='succeeded' or camera.roi_name not in {z['name'] for z in result.roi_snapshot['zones']}:
            raise HTTPException(422,'Camera must reference a completed analysis and its stored ROI')
        video=db.get(Video,job.video_id)
        if camera.registered_camera_id and (db.get(Camera,camera.registered_camera_id) is None or video.camera_id!=camera.registered_camera_id):
            raise HTTPException(422,'Registered camera must match the source video camera')
        for gate in payload.configuration.corridors:
            if gate.camera_id==camera.camera_id and any(x>result.width or y>result.height for x,y in [*gate.boundary,gate.center_side]):
                raise HTTPException(422,'Gate pixels exceed source resolution')
        snapshots[str(camera.job_id)]=result.roi_snapshot
        camera_sources[camera.camera_id]={'video_id':str(job.video_id),'analysis_job_id':str(job.id),
            'result_fingerprint':result.fingerprint,'analysis_mode':job.analysis_mode,
            'video_recorded_at':video.recorded_at.isoformat() if video.recorded_at else None}
    configuration['mapping_roi_snapshots'] = snapshots
    configuration['camera_sources'] = camera_sources
    row = VenueRevision(venue_id=venue_id, revision=payload.base_revision + 1, configuration=configuration)
    db.add(row)
    try:
        db.commit()
        db.refresh(row)
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, 'Concurrent save; reload configuration') from None
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(503, 'Venue storage unavailable') from None
    return view(row)


@router.get('/{venue_id}/festival-flow')
def festival_flow(venue_id: str):
    """축제 전체 흐름 지도: 카메라(게이트)별 방향·시간 오프셋 설정(configs/festival_flow.json)과
    카메라별 경계선 통과 수(uploads/results/<job>/gate_flow.json, python -m app.gate_flow 로 생성)를 함께 준다.
    구역 인원 = 시작 인원 + 누적 유입 - 누적 유출 계산과 10/30/60초 예측은 화면에서 한다."""
    import json
    from pathlib import Path
    from app.services.vision_results import BASE_DIR
    config_path = Path(__file__).resolve().parents[1] / 'configs' / 'festival_flow.json'
    if not config_path.is_file():
        raise HTTPException(404, 'Festival flow is not configured')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('venue_id') != venue_id:
        raise HTTPException(404, 'Festival flow is not configured for this venue')
    phase_cameras = [c for ph in config.get('phases', {}).values() for c in ph.get('cameras', [])]
    for camera in config['cameras'] + phase_cameras:
        camera['gate'] = None
        job_id = camera.get('job_id')
        if job_id:
            gate = (BASE_DIR / 'uploads/results' / str(UUID(job_id)) / 'gate_flow.json')
            if gate.is_file():
                camera['gate'] = json.loads(gate.read_text(encoding='utf-8'))
    return config
