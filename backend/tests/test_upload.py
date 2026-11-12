from unittest.mock import Mock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.database import get_db
from app.main import app
from app.models import Video
from routers import videos


@pytest.fixture
def client(db, tmp_path, monkeypatch):
    monkeypatch.setattr(videos, "UPLOAD_DIR", tmp_path)
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.mark.integration
def test_upload_preserves_response_and_persists_metadata(client, db, tmp_path):
    response = client.post("/api/videos/upload", files={"file": ("test.MP4", b"fixture", "video/mp4")})
    assert response.status_code == 201
    payload = response.json()
    assert set(payload) == {"video_id", "filename", "saved_filename", "file_size", "status", "data_source"}
    row = db.get(Video, UUID(hex=payload["video_id"]))
    assert row.filename == payload["filename"] == "test.MP4"
    assert row.saved_filename == payload["saved_filename"]
    assert row.file_size == payload["file_size"] == 7
    assert row.status == payload["status"] == "uploaded"
    assert row.data_source == payload["data_source"] == "user_upload"
    assert row.storage_path == f"uploads/videos/{row.saved_filename}"
    assert row.recorded_at is None and row.duration_ms is None
    assert (tmp_path / row.saved_filename).read_bytes() == b"fixture"


@pytest.mark.integration
@pytest.mark.parametrize("filename", ["a.mov", "a.avi", "a.webm"])
def test_allowed_formats_do_not_require_browser_mime(client, filename):
    assert client.post("/api/videos/upload", files={"file": (filename, b"fixture", "application/octet-stream")}).status_code == 201


@pytest.mark.integration
def test_invalid_extension_and_oversize_do_not_save_files(client, tmp_path, monkeypatch):
    assert client.post("/api/videos/upload", files={"file": ("a.exe", b"x")}).status_code == 400
    monkeypatch.setattr(videos, "MAX_FILE_SIZE", 3)
    assert client.post("/api/videos/upload", files={"file": ("a.mp4", b"four")}).status_code == 413
    assert list(tmp_path.iterdir()) == []


@pytest.mark.integration
def test_db_failure_returns_safe_error_and_removes_file(client, tmp_path):
    broken_db = Mock()
    broken_db.commit.side_effect = SQLAlchemyError("private database details")
    app.dependency_overrides[get_db] = lambda: broken_db
    response = client.post("/api/videos/upload", files={"file": ("a.mp4", b"fixture")})
    assert response.status_code == 503
    assert "private" not in response.text
    assert list(tmp_path.iterdir()) == []
    broken_db.rollback.assert_called_once()


@pytest.mark.integration
def test_existing_dashboard_and_health_endpoints(client):
    assert client.get("/health").status_code == 200
    assert client.get("/api/database/health").json()["connected"] is True
    response = client.get("/api/dashboard/summary")
    assert response.status_code == 200
    assert response.json()["data_source"] == "mock"
