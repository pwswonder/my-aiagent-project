from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from paper_agent_v2 import api
from paper_agent_v2.config import Settings
from paper_agent_v2.db import Base, get_session
from paper_agent_v2.models import AnalysisRun, Document


def test_upload_returns_durable_run_and_uses_uuid_storage(tmp_path: Path, monkeypatch) -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    settings = Settings(
        database_url="sqlite://",
        storage_root=tmp_path / "storage",
        max_upload_bytes=1_024,
    )
    settings.ensure_directories()
    monkeypatch.setattr(api, "get_settings", lambda: settings)

    def session_override():
        with Session(engine) as session:
            yield session

    app = api.create_app()
    app.dependency_overrides[get_session] = session_override
    client = TestClient(app)
    response = client.post(
        "/api/v2/documents",
        files={"file": ("../../unsafe.pdf", b"%PDF-1.4\n%%EOF", "application/pdf")},
    )
    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    with Session(engine) as session:
        document = session.get(Document, payload["document_id"])
        run = session.get(AnalysisRun, payload["analysis_run_id"])
        assert document is not None and run is not None
        assert Path(document.storage_path).name == "source.pdf"
        assert Path(document.storage_path).parent.name == document.id
        assert session.scalar(select(Document).where(Document.id == document.id)) is not None
