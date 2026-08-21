from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_agent_v2.config import get_settings
from paper_agent_v2.db import SessionLocal, get_session
from paper_agent_v2.ir import ModelGraphSpec, SpecStatus
from paper_agent_v2.models import (
    AnalysisRun,
    ArchitectureSpecRecord,
    Artifact,
    Chunk,
    Document,
    Generation,
    QATurn,
    now,
)
from paper_agent_v2.parser import PaperChunk, PdfValidationError, validate_pdf
from paper_agent_v2.providers.openai_provider import build_provider
from paper_agent_v2.retrieval import HybridRetriever, RetrievalHit, postgres_rrf_chunk_ids
from paper_agent_v2.schemas import (
    Citation,
    DocumentAccepted,
    GenerationAccepted,
    GenerationResponse,
    QAResponse,
    QuestionRequest,
    RunResponse,
    SpecResponse,
)

router = APIRouter(prefix="/api/v2", tags=["paper-agent-v2"])


def _get_or_404(session: Session, model: type, identifier: str):
    value = session.get(model, identifier)
    if value is None:
        raise HTTPException(status_code=404, detail=f"{model.__name__} not found")
    return value


@router.post("/documents", response_model=DocumentAccepted, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(file: UploadFile = File(...), session: Session = Depends(get_session)) -> DocumentAccepted:
    settings = get_settings()
    filename = Path(file.filename or "paper.pdf").name
    if Path(filename).suffix.lower() != ".pdf" or file.content_type not in {
        "application/pdf",
        "application/octet-stream",
    }:
        raise HTTPException(status_code=415, detail="only PDF uploads are accepted")

    document_id = str(uuid4())
    document_dir = (settings.storage_root / "documents" / document_id).resolve()
    document_dir.mkdir(parents=True, exist_ok=False)
    path = document_dir / "source.pdf"
    digest = hashlib.sha256()
    written = 0
    try:
        with path.open("xb") as destination:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="PDF exceeds upload limit")
                digest.update(chunk)
                destination.write(chunk)
        validate_pdf(path, settings.max_upload_bytes)
    except HTTPException:
        path.unlink(missing_ok=True)
        document_dir.rmdir()
        raise
    except PdfValidationError as exc:
        path.unlink(missing_ok=True)
        document_dir.rmdir()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()

    document = Document(
        id=document_id,
        filename=filename,
        storage_path=str(path),
        sha256=digest.hexdigest(),
        status="queued",
    )
    run = AnalysisRun(
        document_id=document.id,
        kind="analysis",
        max_attempts=settings.max_job_attempts,
        event_log=[{"sequence": 1, "stage": "queued", "progress": 0, "message": "analysis queued"}],
    )
    session.add_all([document, run])
    session.commit()
    return DocumentAccepted(document_id=document.id, analysis_run_id=run.id, status=run.status)


@router.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: str, session: Session = Depends(get_session)) -> AnalysisRun:
    return _get_or_404(session, AnalysisRun, run_id)


@router.get("/runs/{run_id}/events")
def stream_run_events(run_id: str, session: Session = Depends(get_session)) -> StreamingResponse:
    _get_or_404(session, AnalysisRun, run_id)

    async def events():
        delivered = 0
        while True:
            with SessionLocal() as event_session:
                run = event_session.get(AnalysisRun, run_id)
                if run is None:
                    yield 'event: error\ndata: {"detail":"run deleted"}\n\n'
                    return
                current = list(run.event_log or [])
                for item in current[delivered:]:
                    yield f"id: {item.get('sequence', delivered + 1)}\nevent: progress\ndata: {json.dumps(item)}\n\n"
                delivered = len(current)
                if run.status in {"completed", "failed", "cancelled"}:
                    yield f"event: done\ndata: {json.dumps({'status': run.status})}\n\n"
                    return
            await asyncio.sleep(1)

    return StreamingResponse(events(), media_type="text/event-stream")


def _latest_spec(session: Session, document_id: str) -> ArchitectureSpecRecord:
    value = session.scalar(
        select(ArchitectureSpecRecord)
        .where(ArchitectureSpecRecord.document_id == document_id)
        .order_by(ArchitectureSpecRecord.version.desc())
        .limit(1)
    )
    if value is None:
        raise HTTPException(status_code=404, detail="Architecture IR not found")
    return value


@router.get("/documents/{document_id}/spec", response_model=SpecResponse)
def get_spec(document_id: str, session: Session = Depends(get_session)) -> SpecResponse:
    _get_or_404(session, Document, document_id)
    record = _latest_spec(session, document_id)
    return SpecResponse(
        document_id=document_id,
        version=record.version,
        status=record.status,
        spec=ModelGraphSpec.model_validate(record.spec_json),
    )


@router.patch("/documents/{document_id}/spec", response_model=SpecResponse)
def patch_spec(document_id: str, spec: ModelGraphSpec, session: Session = Depends(get_session)) -> SpecResponse:
    document = _get_or_404(session, Document, document_id)
    latest = _latest_spec(session, document_id)
    spec.status = SpecStatus.DRAFT
    version = latest.version + 1
    record = ArchitectureSpecRecord(
        document_id=document_id,
        version=version,
        status=spec.status.value,
        spec_json=spec.model_dump(mode="json"),
    )
    document.status = "needs_review"
    session.add(record)
    session.commit()
    return SpecResponse(document_id=document_id, version=version, status=record.status, spec=spec)


@router.post("/documents/{document_id}/spec/{version}/approve", response_model=SpecResponse)
def approve_spec(document_id: str, version: int, session: Session = Depends(get_session)) -> SpecResponse:
    document = _get_or_404(session, Document, document_id)
    record = session.scalar(
        select(ArchitectureSpecRecord).where(
            ArchitectureSpecRecord.document_id == document_id,
            ArchitectureSpecRecord.version == version,
        )
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Architecture IR version not found")
    spec = ModelGraphSpec.model_validate(record.spec_json)
    try:
        spec.approve()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    record.status = "approved"
    record.spec_json = spec.model_dump(mode="json")
    record.approved_at = now()
    document.status = "spec_approved"
    session.commit()
    return SpecResponse(document_id=document_id, version=version, status=record.status, spec=spec)


@router.post(
    "/documents/{document_id}/generations",
    response_model=GenerationAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_generation(document_id: str, session: Session = Depends(get_session)) -> GenerationAccepted:
    _get_or_404(session, Document, document_id)
    spec_record = _latest_spec(session, document_id)
    if spec_record.status != "approved":
        raise HTTPException(status_code=409, detail="latest Architecture IR is not approved")
    generation = Generation(document_id=document_id, spec_id=spec_record.id)
    session.add(generation)
    session.flush()
    run = AnalysisRun(
        document_id=document_id,
        generation_id=generation.id,
        kind="generation",
        max_attempts=get_settings().max_job_attempts,
        event_log=[{"sequence": 1, "stage": "queued", "progress": 0, "message": "generation queued"}],
    )
    session.add(run)
    session.commit()
    return GenerationAccepted(generation_id=generation.id, run_id=run.id, status=run.status)


@router.get("/generations/{generation_id}", response_model=GenerationResponse)
def get_generation(generation_id: str, session: Session = Depends(get_session)) -> GenerationResponse:
    generation = _get_or_404(session, Generation, generation_id)
    artifact_id = session.scalar(select(Artifact.id).where(Artifact.generation_id == generation_id).limit(1))
    return GenerationResponse.model_validate(
        {**GenerationResponse.model_validate(generation).model_dump(), "artifact_id": artifact_id}
    )


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: str, session: Session = Depends(get_session)) -> FileResponse:
    artifact = _get_or_404(session, Artifact, artifact_id)
    settings = get_settings()
    path = Path(artifact.storage_path).resolve()
    root = (settings.storage_root / "artifacts").resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="artifact file is unavailable")
    return FileResponse(path, filename=path.name, media_type="application/zip")


@router.post("/documents/{document_id}/questions", response_model=QAResponse)
def ask_question(document_id: str, request: QuestionRequest, session: Session = Depends(get_session)) -> QAResponse:
    _get_or_404(session, Document, document_id)
    rows = session.scalars(select(Chunk).where(Chunk.document_id == document_id)).all()
    chunks = [
        PaperChunk(
            id=row.id,
            page=row.page,
            section=row.section,
            kind=row.kind,
            text=row.text,
            bbox=(tuple(row.bbox[:4]) if row.bbox and len(row.bbox) >= 4 else None),  # type: ignore[arg-type]
        )
        for row in rows
    ]
    if not chunks:
        raise HTTPException(status_code=409, detail="document analysis is not complete")
    provider = build_provider(get_settings())
    query_embedding = provider.embed([request.question])[0]
    embeddings = [row.embedding or [] for row in rows]
    use_dense = all(embeddings) and all(len(item) == len(query_embedding) for item in embeddings)
    if session.bind and session.bind.dialect.name == "postgresql" and use_dense:
        ranked_ids = postgres_rrf_chunk_ids(
            session, document_id, request.question, query_embedding, limit=request.limit
        )
        by_id = {chunk.id: chunk for chunk in chunks}
        hits = [
            RetrievalHit(chunk=by_id[chunk_id], score=1.0 / (60 + rank))
            for rank, chunk_id in enumerate(ranked_ids, start=1)
            if chunk_id in by_id
        ]
    else:
        hits = HybridRetriever(chunks, embeddings if use_dense else None).search(
            request.question, query_embedding=query_embedding if use_dense else None, limit=request.limit
        )
    query_terms = {term for term in re.findall(r"[A-Za-z0-9가-힣_]+", request.question.lower()) if len(term) > 1}
    supported = [
        hit for hit in hits if query_terms.intersection(re.findall(r"[A-Za-z0-9가-힣_]+", hit.chunk.text.lower()))
    ]
    debug = {
        "retrieved": len(hits),
        "supported": len(supported),
        "chunk_ids": [hit.chunk.id for hit in hits],
        "dense_used": use_dense,
    }
    if not supported:
        response = QAResponse(
            answer=None,
            citations=[],
            answerability="insufficient_evidence",
            retrieval_debug=debug,
        )
    else:
        context = [
            {
                "chunk_id": hit.chunk.id,
                "page": hit.chunk.page,
                "section": hit.chunk.section,
                "text": hit.chunk.text,
            }
            for hit in supported
        ]
        answer = provider.generate_text(
            instructions=(
                "Answer only from the supplied paper excerpts. Cite chunk ids in square brackets. "
                "If the excerpts do not support the answer, output INSUFFICIENT_EVIDENCE."
            ),
            prompt=json.dumps({"question": request.question, "evidence": context}, ensure_ascii=False),
        )
        if "INSUFFICIENT_EVIDENCE" in answer:
            response = QAResponse(
                answer=None, citations=[], answerability="insufficient_evidence", retrieval_debug=debug
            )
        else:
            response = QAResponse(
                answer=answer,
                citations=[
                    Citation(
                        page=hit.chunk.page,
                        section=hit.chunk.section,
                        chunk_id=hit.chunk.id,
                        evidence=hit.chunk.text[:280],
                    )
                    for hit in supported
                ],
                answerability="answerable",
                retrieval_debug=debug,
            )
    session.add(
        QATurn(
            document_id=document_id,
            question=request.question,
            answer=response.answer,
            answerability=response.answerability,
            citations=[item.model_dump() for item in response.citations],
            retrieval_debug=response.retrieval_debug,
        )
    )
    session.commit()
    return response


def create_app() -> FastAPI:
    app = FastAPI(title="AI Paper Agent V2", version="2.0.0a1")
    app.include_router(router)
    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("paper_agent_v2.api:app", host="0.0.0.0", port=8000, reload=False)
