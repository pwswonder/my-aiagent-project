from __future__ import annotations

import asyncio
import hashlib
import json
import re
import zipfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from paper_agent_v2.config import get_settings
from paper_agent_v2.db import SessionLocal, get_session
from paper_agent_v2.ir import Assumption, ModelGraphSpec, SpecStatus
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
    ApproveSpecRequest,
    ArtifactPreview,
    Citation,
    DocumentAccepted,
    DocumentHistoryItem,
    DocumentWorkspace,
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


def _latest_run(session: Session, document_id: str, kind: str) -> AnalysisRun | None:
    return session.scalar(
        select(AnalysisRun)
        .where(AnalysisRun.document_id == document_id, AnalysisRun.kind == kind)
        .order_by(AnalysisRun.created_at.desc())
        .limit(1)
    )


def _generation_response(session: Session, generation: Generation) -> GenerationResponse:
    artifact_id = session.scalar(select(Artifact.id).where(Artifact.generation_id == generation.id).limit(1))
    return GenerationResponse.model_validate(
        {**GenerationResponse.model_validate(generation).model_dump(), "artifact_id": artifact_id}
    )


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
    # Flush the parent first. Without an ORM relationship SQLAlchemy is free to
    # order these independent pending objects with the run first, which violates
    # the PostgreSQL foreign key from runs.document_id to documents.id.
    session.add(document)
    session.flush()
    session.add(run)
    session.commit()
    return DocumentAccepted(document_id=document.id, analysis_run_id=run.id, status=run.status)


@router.get("/documents", response_model=list[DocumentHistoryItem])
def list_documents(session: Session = Depends(get_session)) -> list[DocumentHistoryItem]:
    documents = session.scalars(
        select(Document).where(Document.status != "cancelled").order_by(Document.created_at.desc())
    ).all()
    items = []
    for document in documents:
        run = _latest_run(session, document.id, "analysis")
        items.append(
            DocumentHistoryItem(
                id=document.id,
                filename=document.filename,
                title=document.title,
                status=document.status,
                created_at=document.created_at,
                analysis_run_id=run.id if run else None,
                analysis_status=run.status if run else None,
                analysis_stage=run.stage if run else None,
                analysis_progress=run.progress if run else 0,
            )
        )
    return items


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, session: Session = Depends(get_session)) -> Response:
    """Soft-delete a paper and cancel work that has not finished yet."""
    document = _get_or_404(session, Document, document_id)
    document.status = "cancelled"
    active_runs = session.scalars(
        select(AnalysisRun).where(
            AnalysisRun.document_id == document_id,
            AnalysisRun.status.in_(["queued", "running"]),
        )
    ).all()
    for run in active_runs:
        events = list(run.event_log or [])
        events.append(
            {
                "sequence": len(events) + 1,
                "stage": "cancelled",
                "progress": run.progress,
                "message": "document removed from history",
            }
        )
        run.event_log = events
        run.status = "cancelled"
        run.stage = "cancelled"
        run.locked_at = None
    active_generations = session.scalars(
        select(Generation).where(
            Generation.document_id == document_id,
            Generation.status.in_(["queued", "running"]),
        )
    ).all()
    for generation in active_generations:
        generation.status = "cancelled"
    session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/documents/{document_id}/reanalyze", response_model=DocumentAccepted, status_code=status.HTTP_202_ACCEPTED
)
def reanalyze_document(document_id: str, session: Session = Depends(get_session)) -> DocumentAccepted:
    document = _get_or_404(session, Document, document_id)
    settings = get_settings()
    run = AnalysisRun(
        document_id=document.id,
        kind="analysis",
        max_attempts=settings.max_job_attempts,
        event_log=[{"sequence": 1, "stage": "queued", "progress": 0, "message": "analysis queued"}],
    )
    document.status = "queued"
    session.add(run)
    session.commit()
    return DocumentAccepted(document_id=document.id, analysis_run_id=run.id, status=run.status)


@router.get("/documents/{document_id}/workspace", response_model=DocumentWorkspace)
def get_document_workspace(document_id: str, session: Session = Depends(get_session)) -> DocumentWorkspace:
    document = _get_or_404(session, Document, document_id)
    analysis_run = _latest_run(session, document_id, "analysis")
    spec_record = session.scalar(
        select(ArchitectureSpecRecord)
        .where(ArchitectureSpecRecord.document_id == document_id)
        .order_by(ArchitectureSpecRecord.version.desc())
        .limit(1)
    )
    spec_response = None
    if spec_record:
        spec_response = SpecResponse(
            document_id=document_id,
            version=spec_record.version,
            status=spec_record.status,
            spec=ModelGraphSpec.model_validate(spec_record.spec_json),
        )
    generation = session.scalar(
        select(Generation).where(Generation.document_id == document_id).order_by(Generation.created_at.desc()).limit(1)
    )
    generation_response = _generation_response(session, generation) if generation else None
    generation_run = (
        session.scalar(
            select(AnalysisRun)
            .where(AnalysisRun.generation_id == generation.id)
            .order_by(AnalysisRun.created_at.desc())
            .limit(1)
        )
        if generation
        else None
    )
    qa_history = session.scalars(
        select(QATurn).where(QATurn.document_id == document_id).order_by(QATurn.created_at)
    ).all()
    summary = str((analysis_run.payload or {}).get("summary") or "") if analysis_run else ""
    if not summary:
        fallback_chunks = session.scalars(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.page, Chunk.id).limit(3)
        ).all()
        summary = "\n\n".join(chunk.text for chunk in fallback_chunks)[:4_000]
    return DocumentWorkspace(
        document_id=document.id,
        filename=document.filename,
        title=document.title,
        status=document.status,
        created_at=document.created_at,
        summary=summary or None,
        analysis_run=analysis_run,
        spec=spec_response,
        generation=generation_response,
        generation_run=generation_run,
        qa_history=qa_history,
    )


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
        raise HTTPException(status_code=404, detail="model structure not found")
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
def approve_spec(
    document_id: str,
    version: int,
    request: ApproveSpecRequest | None = None,
    session: Session = Depends(get_session),
) -> SpecResponse:
    document = _get_or_404(session, Document, document_id)
    record = session.scalar(
        select(ArchitectureSpecRecord).where(
            ArchitectureSpecRecord.document_id == document_id,
            ArchitectureSpecRecord.version == version,
        )
    )
    if record is None:
        raise HTTPException(status_code=404, detail="model structure version not found")
    spec = ModelGraphSpec.model_validate(record.spec_json)
    if request and request.accept_blocking_as_assumptions:
        _accept_blocking_as_assumptions(spec)
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


def _accept_blocking_as_assumptions(spec: ModelGraphSpec) -> None:
    existing_assumptions = {item.field for item in spec.assumptions}
    for item in spec.unresolved:
        if not item.blocking:
            continue
        if item.field not in existing_assumptions:
            spec.assumptions.append(
                Assumption(
                    field=item.field,
                    value="use the reviewed model structure and registry defaults",
                    reason=(
                        "The paper or a verified official-code snapshot did not fully specify this detail. "
                        "The user explicitly accepted the reviewed model structure defaults."
                    ),
                    confidence=0.35,
                )
            )
        item.blocking = False


def _qa_candidates(
    hits: list[RetrievalHit], question: str, *, dense_used: bool
) -> tuple[list[RetrievalHit], str]:
    """Keep semantic results across languages; require lexical support only without embeddings."""
    if dense_used:
        return hits, "semantic_rrf"
    query_terms = {
        term for term in re.findall(r"[A-Za-z0-9가-힣_]+", question.lower()) if len(term) > 1
    }
    supported = [
        hit
        for hit in hits
        if query_terms.intersection(re.findall(r"[A-Za-z0-9가-힣_]+", hit.chunk.text.lower()))
    ]
    return supported, "lexical_overlap"


def _augment_architecture_hits(
    hits: list[RetrievalHit],
    chunks: list[PaperChunk],
    spec: ModelGraphSpec | None,
    question: str,
    *,
    evidence_limit: int = 6,
) -> list[RetrievalHit]:
    """Add IR-grounding evidence for broad model/method questions that embeddings can underspecify."""
    if spec is None or not re.search(
        r"(모델|구조|아키텍처|방법|제안|model|architecture|method|propos)", question.lower()
    ):
        return hits
    evidence_ids = []
    for node in spec.topological_nodes():
        evidence_ids.extend(node.evidence_ids)
    by_id = {chunk.id: chunk for chunk in chunks}
    grounded = []
    for evidence_id in dict.fromkeys(evidence_ids):
        if evidence_id in by_id:
            grounded.append(RetrievalHit(chunk=by_id[evidence_id], score=1.0))
        if len(grounded) >= evidence_limit:
            break
    existing = {hit.chunk.id for hit in grounded}
    return grounded + [hit for hit in hits if hit.chunk.id not in existing]


def _looks_like_bibliography(chunk: PaperChunk) -> bool:
    return chunk.section == "references" or bool(re.match(r"^\s*\[\d+\]\s", chunk.text))


def _query_anchor_hits(chunks: list[PaperChunk], question: str) -> list[RetrievalHit]:
    """Add deterministic paper regions for common broad questions.

    Dense retrieval can rank a section heading without the paragraph directly
    below it. These anchors do not answer the question themselves; they only
    make the relevant paper excerpts available to the answer model.
    """
    lowered = question.lower()
    candidates: list[tuple[int, PaperChunk]] = []
    if re.search(r"(저자|author|쓴 사람|연구진)", lowered):
        candidates = [(100 - index, chunk) for index, chunk in enumerate(chunks) if chunk.page == 1]
    elif re.search(r"(한계|제약|약점|limitation|drawback|future work)", lowered):
        for chunk in chunks:
            text_lower = chunk.text.lower()
            score = 0
            if chunk.section in {"limitations", "discussion", "conclusion"}:
                score += 10
            if re.search(r"limitation|future work|however|remain|fails?|challenge", text_lower):
                score += 6
            if score and len(chunk.text) >= 40 and not _looks_like_bibliography(chunk):
                candidates.append((score, chunk))
    elif re.search(r"(모델|구조|아키텍처|방법|제안|특징|model|architecture|method|propos|framework)", lowered):
        for chunk in chunks:
            text_lower = chunk.text.lower()
            score = 0
            if chunk.section in {"method", "model", "architecture", "approach"}:
                score += 10
            if re.search(r"proposed|framework|architecture|operates in|main contributions", text_lower):
                score += 6
            if score and len(chunk.text) >= 40 and not _looks_like_bibliography(chunk):
                candidates.append((score, chunk))
    candidates.sort(key=lambda item: (-item[0], item[1].page, item[1].id))
    return [RetrievalHit(chunk=chunk, score=float(score)) for score, chunk in candidates[:5]]


def _expand_qa_context(
    hits: list[RetrievalHit], chunks: list[PaperChunk], question: str, *, limit: int = 20
) -> list[RetrievalHit]:
    """Expand layout-block hits with nearby paragraphs while retaining citations."""
    ordered = sorted(chunks, key=lambda item: (item.page, item.id))
    positions = {chunk.id: index for index, chunk in enumerate(ordered)}
    seeds = _query_anchor_hits(chunks, question) + hits
    expanded: list[RetrievalHit] = []
    seen: set[str] = set()
    for seed in seeds:
        position = positions.get(seed.chunk.id)
        if position is None:
            continue
        neighbourhood = ordered[max(0, position - 2) : position + 4]
        for chunk in [seed.chunk, *neighbourhood]:
            if chunk.id in seen or _looks_like_bibliography(chunk):
                continue
            if len(chunk.text.strip()) < 10 and chunk.page != 1:
                continue
            seen.add(chunk.id)
            expanded.append(RetrievalHit(chunk=chunk, score=seed.score))
            if len(expanded) >= limit:
                return expanded
    return expanded


@router.post(
    "/documents/{document_id}/generations",
    response_model=GenerationAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_generation(document_id: str, session: Session = Depends(get_session)) -> GenerationAccepted:
    _get_or_404(session, Document, document_id)
    spec_record = _latest_spec(session, document_id)
    if spec_record.status != "approved":
        raise HTTPException(status_code=409, detail="latest model structure is not approved")
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
    return _generation_response(session, generation)


def _latest_generation_source(generation_id: str) -> str | None:
    settings = get_settings()
    generation_root = (settings.storage_root / "artifacts" / generation_id).resolve()
    artifacts_root = (settings.storage_root / "artifacts").resolve()
    if artifacts_root not in generation_root.parents or not generation_root.is_dir():
        return None
    attempts = [
        path
        for path in generation_root.iterdir()
        if path.is_dir() and re.fullmatch(r"attempt-(\d+)", path.name)
    ]
    attempts.sort(key=lambda path: int(path.name.removeprefix("attempt-")), reverse=True)
    for attempt in attempts:
        candidates = [attempt / "model.py", *attempt.glob("*/model.py")]
        for candidate in candidates:
            model_path = candidate.resolve()
            if attempt.resolve() not in model_path.parents or not model_path.is_file():
                continue
            if model_path.stat().st_size > 500_000:
                continue
            return model_path.read_text(encoding="utf-8", errors="replace")
    return None


@router.get("/generations/{generation_id}/preview", response_model=ArtifactPreview)
def preview_generation(generation_id: str, session: Session = Depends(get_session)) -> ArtifactPreview:
    _get_or_404(session, Generation, generation_id)
    source = _latest_generation_source(generation_id)
    if source is None:
        raise HTTPException(status_code=404, detail="generated model source is unavailable")
    return ArtifactPreview(files={"model.py": source})


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: str, session: Session = Depends(get_session)) -> FileResponse:
    artifact = _get_or_404(session, Artifact, artifact_id)
    settings = get_settings()
    path = Path(artifact.storage_path).resolve()
    root = (settings.storage_root / "artifacts").resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="artifact file is unavailable")
    return FileResponse(path, filename=path.name, media_type="application/zip")


@router.get("/artifacts/{artifact_id}/preview", response_model=ArtifactPreview)
def preview_artifact(artifact_id: str, session: Session = Depends(get_session)) -> ArtifactPreview:
    artifact = _get_or_404(session, Artifact, artifact_id)
    settings = get_settings()
    path = Path(artifact.storage_path).resolve()
    root = (settings.storage_root / "artifacts").resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="artifact file is unavailable")
    allowed = {"model.py", "config.py", "README.md", "validation.json"}
    files = {}
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.filename in allowed and info.file_size <= 200_000:
                    files[info.filename] = archive.read(info).decode("utf-8", errors="replace")
    except (OSError, zipfile.BadZipFile) as exc:
        raise HTTPException(status_code=422, detail="artifact archive is invalid") from exc
    return ArtifactPreview(files=files)


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
    try:
        spec = ModelGraphSpec.model_validate(_latest_spec(session, document_id).spec_json)
    except HTTPException:
        spec = None
    hits = _augment_architecture_hits(hits, chunks, spec, request.question)
    hits = _expand_qa_context(hits, chunks, request.question)
    supported, evidence_gate = _qa_candidates(hits, request.question, dense_used=use_dense)
    debug = {
        "retrieved": len(hits),
        "supported": len(supported),
        "chunk_ids": [hit.chunk.id for hit in hits],
        "dense_used": use_dense,
        "evidence_gate": evidence_gate,
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
                "Answer in the same language as the question, using only the supplied paper excerpts. "
                "Cite supporting chunk ids in square brackets. "
                "For limitation questions, distinguish limitations explicitly stated by the authors from "
                "limitations cautiously inferred from the reported scope or conclusion. "
                "If the excerpts truly do not support an answer, output INSUFFICIENT_EVIDENCE."
            ),
            prompt=json.dumps({"question": request.question, "evidence": context}, ensure_ascii=False),
        )
        if "INSUFFICIENT_EVIDENCE" in answer:
            response = QAResponse(
                answer=None, citations=[], answerability="insufficient_evidence", retrieval_debug=debug
            )
        else:
            cited_ids = set(re.findall(r"\[([^\[\]]+)\]", answer))
            cited_hits = [hit for hit in supported if hit.chunk.id in cited_ids] or supported
            response = QAResponse(
                answer=answer,
                citations=[
                    Citation(
                        page=hit.chunk.page,
                        section=hit.chunk.section,
                        chunk_id=hit.chunk.id,
                        evidence=hit.chunk.text[:280],
                    )
                    for hit in cited_hits
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
    app = FastAPI(title="AI Paper Agent", version="2.0.0a1")
    app.include_router(router)
    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("paper_agent_v2.api:app", host="0.0.0.0", port=8000, reload=False)
