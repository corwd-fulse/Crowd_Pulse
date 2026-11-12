from datetime import datetime, timedelta, timezone
from io import StringIO
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import insert, inspect, text, UniqueConstraint
from sqlalchemy.exc import IntegrityError

from app.models import (
    Base, Video, AnalysisJob, CrowdMetric, CrowdPrediction, BottleneckEvent,
)


@pytest.fixture
def job(db):
    uid = uuid4()
    video = Video(
        id=uid, filename="fixture.mp4", saved_filename=f"{uid.hex}.mp4",
        storage_path=f"uploads/videos/{uid.hex}.mp4", file_size=5,
    )
    db.add(video)
    db.flush()
    job = AnalysisJob(video_id=video.id)
    db.add(job)
    db.flush()
    return job


def test_time_columns_and_unique_constraints():
    for model, partition in [(CrowdMetric, "analyzed_at"), (CrowdPrediction, "predicted_at")]:
        table = model.__table__
        assert table.c[partition].nullable is False
        assert table.c[partition].type.timezone is True
        assert partition in table.primary_key.columns
        for constraint in table.constraints:
            if isinstance(constraint, UniqueConstraint):
                assert partition in constraint.columns
        for index in table.indexes:
            if index.unique:
                assert partition in index.columns
    for model, columns in [
        (Video, ["recorded_at"]), (CrowdMetric, ["captured_at"]),
        (CrowdPrediction, ["target_captured_at"]),
        (BottleneckEvent, ["captured_start_at", "captured_end_at"]),
    ]:
        for name in columns:
            column = model.__table__.c[name]
            assert column.nullable and column.type.timezone
            assert column.default is None and column.server_default is None


def test_offline_migration_is_additive_and_has_hypertable_ddl():
    output = StringIO()
    config = Config("alembic.ini", output_buffer=output)
    command.upgrade(config, "head", sql=True)
    sql = output.getvalue().upper()
    for name in Base.metadata.tables:
        assert f"CREATE TABLE {name.upper()}" in sql
    assert sql.count("SELECT CREATE_HYPERTABLE(") == 4
    assert "BY_RANGE('ANALYZED_AT'" in sql
    assert "BY_RANGE('PREDICTED_AT'" in sql
    for forbidden in ["DROP TABLE", "TRUNCATE", "CREATE EXTENSION", "ALTER SYSTEM", "DELETE FROM"]:
        assert forbidden not in sql


@pytest.mark.integration
def test_real_database_hypertables_and_primary_keys(db):
    tables = dict(db.execute(text("""
        SELECT hypertable_name, num_dimensions
        FROM timescaledb_information.hypertables WHERE hypertable_schema='public'
    """)).all())
    assert tables["crowd_metrics"] == 1
    assert tables["crowd_predictions"] == 1
    assert not {"videos", "analysis_jobs", "bottleneck_events", "cameras"} & tables.keys()
    dimensions = dict(db.execute(text("""
        SELECT hypertable_name, column_name FROM timescaledb_information.dimensions
        WHERE hypertable_schema='public'
    """)).all())
    assert dimensions["crowd_metrics"] == "analyzed_at"
    assert dimensions["crowd_predictions"] == "predicted_at"
    inspector = inspect(db.connection())
    for table, partition in [("crowd_metrics", "analyzed_at"), ("crowd_predictions", "predicted_at")]:
        assert inspector.get_pk_constraint(table)["constrained_columns"] == ["id", partition]


@pytest.mark.integration
def test_unknown_capture_times_remain_null_and_relative_time_is_separate(db, job):
    now = datetime.now(timezone.utc)
    metric = CrowdMetric(
        analysis_job_id=job.id, analyzed_at=now, video_offset_ms=1234,
        person_count=12, congestion_percent=25,
    )
    prediction = CrowdPrediction(
        analysis_job_id=job.id, predicted_at=now, source_offset_ms=1234,
        target_offset_ms=6234, predicted_person_count=15,
        predicted_congestion_percent=30, model_version="fixture-only",
    )
    event = BottleneckEvent(
        analysis_job_id=job.id, start_offset_ms=1234, severity="medium",
    )
    db.add_all([metric, prediction, event])
    db.flush()
    for row in [metric, prediction, event]:
        db.refresh(row)
    video = db.get(Video, job.video_id)
    assert video.recorded_at is None
    assert video.uploaded_at.tzinfo is not None
    assert metric.captured_at is None and metric.video_offset_ms == 1234
    assert metric.analyzed_at == now
    assert prediction.target_captured_at is None
    assert event.captured_start_at is None and event.captured_end_at is None


@pytest.mark.integration
def test_known_capture_time_and_reanalysis_do_not_share_a_clock(db, job):
    captured = datetime(2020, 1, 1, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    for analyzed_at in [now, now + timedelta(seconds=1)]:
        db.add(CrowdMetric(
            analysis_job_id=job.id, analyzed_at=analyzed_at,
            video_offset_ms=500, captured_at=captured + timedelta(milliseconds=500),
            person_count=1, congestion_percent=0,
        ))
    db.flush()


@pytest.mark.integration
@pytest.mark.parametrize("model,changes", [
    (CrowdMetric, {"video_offset_ms": -1}),
    (CrowdMetric, {"person_count": -1}),
    (CrowdMetric, {"congestion_percent": 101}),
    (CrowdMetric, {"analyzed_at": None}),
    (CrowdMetric, {"density": -1}),
    (CrowdPrediction, {"target_offset_ms": 0}),
    (CrowdPrediction, {"predicted_person_count": -1}),
    (CrowdPrediction, {"predicted_congestion_percent": -1}),
    (CrowdPrediction, {"confidence": 1.1}),
    (CrowdPrediction, {"predicted_at": None}),
    (BottleneckEvent, {"end_offset_ms": -1}),
    (BottleneckEvent, {"severity": "unknown"}),
    (BottleneckEvent, {"captured_end_at": datetime.now(timezone.utc)}),
    (AnalysisJob, {"status": "unknown"}),
])
def test_database_rejects_invalid_values(db, job, model, changes):
    now = datetime.now(timezone.utc)
    defaults = {
        CrowdMetric: dict(analysis_job_id=job.id, analyzed_at=now, video_offset_ms=0, person_count=1, congestion_percent=0),
        CrowdPrediction: dict(analysis_job_id=job.id, predicted_at=now, source_offset_ms=0, target_offset_ms=1000, predicted_person_count=1, predicted_congestion_percent=0, model_version="fixture"),
        BottleneckEvent: dict(analysis_job_id=job.id, start_offset_ms=0, severity="low"),
        AnalysisJob: dict(video_id=job.video_id),
    }
    with pytest.raises(IntegrityError):
        with db.begin_nested():
            db.execute(insert(model).values(**(defaults[model] | changes)))


@pytest.mark.integration
def test_database_enforces_fk_and_full_hypertable_primary_key(db, job):
    values = dict(
        id=uuid4(), analysis_job_id=job.id, analyzed_at=datetime.now(timezone.utc),
        video_offset_ms=0, person_count=1, congestion_percent=0,
    )
    db.execute(insert(CrowdMetric).values(**values))
    with pytest.raises(IntegrityError):
        with db.begin_nested():
            db.execute(insert(CrowdMetric).values(**values))
    with pytest.raises(IntegrityError):
        with db.begin_nested():
            db.execute(insert(CrowdMetric).values(**(values | {"id": uuid4(), "analysis_job_id": uuid4()})))


@pytest.mark.integration
def test_repeated_upgrade_preserves_existing_row(db, job):
    config = Config("alembic.ini")
    config.attributes["connection"] = db.connection()
    command.upgrade(config, "head")
    db.expire_all()
    assert db.get(AnalysisJob, job.id).video_id == job.video_id
