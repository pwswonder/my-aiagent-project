from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from paper_agent_v2 import api
from paper_agent_v2.config import Settings
from paper_agent_v2.db import Base, get_session
from paper_agent_v2.ir import ModelGraphSpec, NodeSpec, TensorSpec, UnresolvedItem
from paper_agent_v2.jobs import _auto_approve, preserve_previous_spec_after_failed_refresh
from paper_agent_v2.models import AnalysisRun, Document
from paper_agent_v2.parser import PaperChunk
from paper_agent_v2.retrieval import RetrievalHit


def test_dense_qa_candidates_do_not_require_cross_language_token_overlap() -> None:
    hit = RetrievalHit(
        chunk=PaperChunk(
            id="page-1",
            page=1,
            section="front_matter",
            kind="paragraph",
            text="The authors are Wenhai Wang and Enze Xie.",
        ),
        score=1.0,
    )

    dense, dense_gate = api._qa_candidates([hit], "이 논문의 저자는 누구야?", dense_used=True)
    lexical, lexical_gate = api._qa_candidates([hit], "이 논문의 저자는 누구야?", dense_used=False)

    assert dense == [hit]
    assert dense_gate == "semantic_rrf"
    assert lexical == []
    assert lexical_gate == "lexical_overlap"


def test_user_can_accept_blocking_items_as_explicit_assumptions() -> None:
    spec = ModelGraphSpec(
        name="Example",
        task="classification",
        inputs=[TensorSpec(name="input", shape=["B", 4])],
        nodes=[
            NodeSpec(
                id="projection",
                op="Linear",
                inputs=["input"],
                output="logits",
                params={"in_features": 4, "out_features": 2},
                evidence_ids=["paper-method"],
            )
        ],
        outputs=[TensorSpec(name="logits", shape=["B", 2])],
        evidence=[
            {
                "id": "paper-method",
                "source_type": "pdf",
                "quote": "A linear classifier produces two logits.",
                "page": 2,
            }
        ],
        unresolved=[UnresolvedItem(field="projection.bias", question="Use bias?", blocking=True)],
    )

    api._accept_blocking_as_assumptions(spec)
    spec.approve()

    assert spec.status == "approved"
    assert not any(item.blocking for item in spec.unresolved)
    assert any(item.field == "projection.bias" for item in spec.assumptions)


def test_model_question_is_augmented_with_ir_evidence() -> None:
    architecture_chunk = PaperChunk(
        id="paper-method",
        page=4,
        section="method",
        kind="paragraph",
        text="The model has four pyramid stages.",
    )
    unrelated_hit = RetrievalHit(
        chunk=PaperChunk(
            id="references",
            page=12,
            section="references",
            kind="paragraph",
            text="Bibliography",
        ),
        score=0.5,
    )
    spec = ModelGraphSpec(
        name="Example",
        task="classification",
        inputs=[TensorSpec(name="input", shape=["B", 4])],
        nodes=[
            NodeSpec(
                id="projection",
                op="Linear",
                inputs=["input"],
                output="logits",
                params={"in_features": 4, "out_features": 2},
                evidence_ids=["paper-method"],
            )
        ],
        outputs=[TensorSpec(name="logits", shape=["B", 2])],
        evidence=[
            {
                "id": "paper-method",
                "source_type": "pdf",
                "quote": "The model has four pyramid stages.",
                "page": 4,
            }
        ],
    )

    augmented = api._augment_architecture_hits(
        [unrelated_hit], [architecture_chunk, unrelated_hit.chunk], spec, "제안 모델을 설명해줘"
    )

    assert [hit.chunk.id for hit in augmented] == ["paper-method", "references"]


def test_qa_context_expands_a_method_heading_to_following_paragraphs() -> None:
    chunks = [
        PaperChunk("heading", 4, "method", "paragraph", "3. Proposed Method"),
        PaperChunk(
            "overview",
            4,
            "method",
            "paragraph",
            "The proposed HierCore framework clusters semantic embeddings and builds hierarchical memory banks.",
        ),
        PaperChunk("reference", 12, "references", "paragraph", "[1] An unrelated paper."),
    ]

    expanded = api._expand_qa_context(
        [RetrievalHit(chunk=chunks[0], score=1.0)], chunks, "제안 모델의 특징을 알려줘"
    )

    assert "overview" in [hit.chunk.id for hit in expanded]
    assert "reference" not in [hit.chunk.id for hit in expanded]


def test_automatic_flow_records_assumptions_and_approves() -> None:
    spec = ModelGraphSpec(
        name="Example",
        task="classification",
        inputs=[TensorSpec(name="input", shape=["B", 4])],
        nodes=[
            NodeSpec(
                id="projection",
                op="Linear",
                inputs=["input"],
                output="logits",
                params={"in_features": 4, "out_features": 2},
                evidence_ids=["paper-method"],
            )
        ],
        outputs=[TensorSpec(name="logits", shape=["B", 2])],
        evidence=[
            {
                "id": "paper-method",
                "source_type": "pdf",
                "quote": "A linear classifier produces two logits.",
                "page": 2,
            }
        ],
        unresolved=[UnresolvedItem(field="projection.bias", question="Use bias?", blocking=True)],
    )

    _auto_approve(spec)

    assert spec.status == "approved"
    assert spec.unresolved[0].blocking is False
    assert spec.assumptions[0].field == "projection.bias"


def test_generation_preview_returns_latest_attempt_source(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(storage_root=tmp_path / "storage")
    settings.ensure_directories()
    generation_id = "00000000-0000-0000-0000-000000000001"
    attempt_zero = settings.storage_root / "artifacts" / generation_id / "attempt-0"
    attempt_two = settings.storage_root / "artifacts" / generation_id / "attempt-2"
    attempt_zero.mkdir(parents=True)
    attempt_two.mkdir(parents=True)
    (attempt_zero / "model.py").write_text("old source", encoding="utf-8")
    package_dir = attempt_two / "generated-package"
    package_dir.mkdir()
    (package_dir / "model.py").write_text("latest source", encoding="utf-8")
    monkeypatch.setattr(api, "get_settings", lambda: settings)

    assert api._latest_generation_source(generation_id) == "latest source"


def test_failed_reanalysis_preserves_previous_implementable_spec() -> None:
    previous = ModelGraphSpec(
        name="Working",
        task="classification",
        inputs=[TensorSpec(name="input", shape=["B", 4])],
        nodes=[
            NodeSpec(
                id="projection",
                op="Linear",
                inputs=["input"],
                output="logits",
                params={"in_features": 4, "out_features": 2},
                evidence_ids=["paper-method"],
            )
        ],
        outputs=[TensorSpec(name="logits", shape=["B", 2])],
        evidence=[
            {
                "id": "paper-method",
                "source_type": "pdf",
                "quote": "A linear classifier produces two logits.",
                "page": 2,
            }
        ],
    )
    failed = ModelGraphSpec(
        name="Failed",
        task="unknown",
        inputs=[TensorSpec(name="input", shape=["B", "..."])],
        nodes=[],
        outputs=[TensorSpec(name="input", shape=["B", "..."])],
        unresolved=[UnresolvedItem(field="architecture", question="Validation failed")],
    )

    preserved = preserve_previous_spec_after_failed_refresh(failed, previous)

    assert preserved.name == "Working"
    assert len(preserved.nodes) == 1
    assert any(item.field == "latest_analysis_refresh" for item in preserved.unresolved)


def test_upload_returns_durable_run_and_uses_uuid_storage(tmp_path: Path, monkeypatch) -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

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
    history_response = client.get("/api/v2/documents")
    assert history_response.status_code == 200
    assert history_response.json()[0]["id"] == payload["document_id"]
    assert history_response.json()[0]["analysis_run_id"] == payload["analysis_run_id"]
    workspace_response = client.get(f"/api/v2/documents/{payload['document_id']}/workspace")
    assert workspace_response.status_code == 200
    assert workspace_response.json()["analysis_run"]["id"] == payload["analysis_run_id"]
    assert workspace_response.json()["qa_history"] == []
    reanalysis_response = client.post(f"/api/v2/documents/{payload['document_id']}/reanalyze")
    assert reanalysis_response.status_code == 202
    assert reanalysis_response.json()["analysis_run_id"] != payload["analysis_run_id"]
    delete_response = client.delete(f"/api/v2/documents/{payload['document_id']}")
    assert delete_response.status_code == 204
    assert client.get("/api/v2/documents").json() == []
    with Session(engine) as session:
        document = session.get(Document, payload["document_id"])
        run = session.get(AnalysisRun, payload["analysis_run_id"])
        assert document is not None and run is not None
        assert document.status == "cancelled"
        assert run.status == "cancelled"
        assert Path(document.storage_path).name == "source.pdf"
        assert Path(document.storage_path).parent.name == document.id
        assert session.scalar(select(Document).where(Document.id == document.id)) is not None
