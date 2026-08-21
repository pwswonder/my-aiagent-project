from __future__ import annotations

import hashlib
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from paper_agent_v2.analysis import analyze_paper
from paper_agent_v2.config import Settings, get_settings
from paper_agent_v2.generation.custom import synthesize_custom_module, validate_custom_module
from paper_agent_v2.generation.package_writer import write_package
from paper_agent_v2.generation.renderer import render_model
from paper_agent_v2.ir import Assumption, ModelGraphSpec, SpecStatus, UnresolvedItem
from paper_agent_v2.models import (
    AnalysisRun,
    ArchitectureSpecRecord,
    Artifact,
    Chunk,
    Document,
    Evidence,
    ExternalSource,
    Generation,
    PaperSection,
)
from paper_agent_v2.official_code import GitHubSourceResolver
from paper_agent_v2.providers.openai_provider import build_provider
from paper_agent_v2.repair import RestrictedRepairPatch, repair_priority
from paper_agent_v2.sandbox import (
    DockerSandbox,
    FailureCategory,
    RemoteSandbox,
    SandboxResult,
    save_validation_result,
)


def append_event(run: AnalysisRun, stage: str, progress: int, message: str) -> None:
    events = list(run.event_log or [])
    events.append(
        {
            "sequence": len(events) + 1,
            "timestamp": datetime.now(UTC).isoformat(),
            "stage": stage,
            "progress": progress,
            "message": message,
        }
    )
    run.event_log = events
    run.stage = stage
    run.progress = progress


def preserve_previous_spec_after_failed_refresh(
    extracted: ModelGraphSpec, previous: ModelGraphSpec | None
) -> ModelGraphSpec:
    if extracted.nodes or previous is None or not previous.nodes:
        return extracted
    reason = extracted.unresolved[0].question if extracted.unresolved else "Architecture refresh failed validation."
    previous.status = SpecStatus.NEEDS_REVIEW
    if not any(item.field == "latest_analysis_refresh" for item in previous.unresolved):
        previous.unresolved.append(
            UnresolvedItem(
                field="latest_analysis_refresh",
                question=f"The latest refresh was not applied: {reason}",
                blocking=False,
            )
        )
    return previous


def claim_next_run(session: Session) -> AnalysisRun | None:
    statement = select(AnalysisRun).where(AnalysisRun.status == "queued").order_by(AnalysisRun.created_at).limit(1)
    if session.bind and session.bind.dialect.name == "postgresql":
        statement = statement.with_for_update(skip_locked=True)
    run = session.scalar(statement)
    if run is None:
        return None
    run.status = "running"
    run.attempts += 1
    run.locked_at = datetime.now(UTC)
    append_event(run, "claimed", 1, f"worker claimed attempt {run.attempts}")
    session.commit()
    return run


def recover_stale_runs(session: Session, age: timedelta = timedelta(minutes=5)) -> int:
    threshold = datetime.now(UTC) - age
    result = session.execute(
        update(AnalysisRun)
        .where(AnalysisRun.status == "running", AnalysisRun.locked_at < threshold)
        .values(status="queued", stage="recovered", locked_at=None)
    )
    session.commit()
    return int(getattr(result, "rowcount", 0) or 0)


def _record_chunks(session: Session, document: Document, result: Any, provider: Any) -> None:
    texts = [chunk.text for chunk in result.paper.chunks]
    try:
        embeddings = provider.embed(texts)
    except Exception:
        embeddings = [None] * len(texts)
    for index, chunk in enumerate(result.paper.chunks):
        session.merge(
            Chunk(
                id=chunk.id,
                document_id=document.id,
                page=chunk.page,
                section=chunk.section,
                kind=chunk.kind,
                text=chunk.text,
                bbox=list(chunk.bbox) if chunk.bbox else None,
                embedding=embeddings[index] if index < len(embeddings) else None,
            )
        )
    for item in result.spec.evidence:
        session.merge(
            Evidence(
                id=item.id,
                document_id=document.id,
                source_type=item.source_type.value,
                quote=item.quote,
                page=item.page,
                section=item.section,
                url=item.url,
                commit_sha=item.commit_sha,
                chunk_id=item.chunk_id,
            )
        )


def _auto_approve(spec: ModelGraphSpec) -> None:
    """Record paper omissions as assumptions for the default one-click flow."""
    existing = {item.field for item in spec.assumptions}
    for item in spec.unresolved:
        if not item.blocking:
            continue
        if item.field not in existing:
            spec.assumptions.append(
                Assumption(
                    field=item.field,
                    value="use the extracted model structure and registry defaults",
                    reason=(
                        "The paper did not fully specify this detail. The automatic flow records the "
                        "generator's evidence-grounded value as an explicit assumption."
                    ),
                    confidence=0.35,
                )
            )
        item.blocking = False
    spec.approve()


def process_analysis(session: Session, run: AnalysisRun, settings: Settings) -> None:
    document = session.get(Document, run.document_id)
    if document is None:
        raise ValueError("analysis document no longer exists")
    provider = build_provider(settings)
    append_event(run, "parsing", 10, "validating and parsing PDF")
    session.commit()
    render_dir = settings.storage_root / "documents" / document.id / "rendered"
    resolver = GitHubSourceResolver(settings.github_token)
    result = analyze_paper(
        Path(document.storage_path),
        provider,
        max_bytes=settings.max_upload_bytes,
        title_hint=Path(document.filename).stem,
        render_dir=render_dir,
        source_resolver=resolver,
    )
    latest_record = session.scalar(
        select(ArchitectureSpecRecord)
        .where(ArchitectureSpecRecord.document_id == document.id)
        .order_by(ArchitectureSpecRecord.version.desc())
        .limit(1)
    )
    previous_spec = ModelGraphSpec.model_validate(latest_record.spec_json) if latest_record else None
    result.spec = preserve_previous_spec_after_failed_refresh(result.spec, previous_spec)
    append_event(run, "persisting", 70, "saving chunks, evidence and model structure")
    run.payload = {**(run.payload or {}), "summary": result.summary}
    document.sha256 = result.paper.sha256
    paper_title = result.paper.title
    if paper_title.strip().lower() in {"source", "untitled", "document"}:
        paper_title = Path(document.filename).stem
    document.title = paper_title
    if result.spec.name.strip().lower() in {"source", "untitled", "document"}:
        result.spec.name = paper_title
    # Re-analysis replaces searchable material instead of retaining stale
    # chunks and duplicated section/source records from earlier attempts.
    session.execute(delete(Evidence).where(Evidence.document_id == document.id))
    session.execute(delete(PaperSection).where(PaperSection.document_id == document.id))
    session.execute(delete(ExternalSource).where(ExternalSource.document_id == document.id))
    session.execute(delete(Chunk).where(Chunk.document_id == document.id))

    auto_generate = bool(result.spec.nodes)
    if auto_generate:
        _auto_approve(result.spec)
        document.status = "generating"
    else:
        document.status = "needs_review"
    _record_chunks(session, document, result, provider)
    section_pages: dict[str, list[int]] = {}
    for chunk in result.paper.chunks:
        section_pages.setdefault(chunk.section, []).append(chunk.page)
    for name, pages in section_pages.items():
        session.add(
            PaperSection(
                document_id=document.id,
                name=name,
                page_start=min(pages),
                page_end=max(pages),
            )
        )
    version = (latest_record.version if latest_record else 0) + 1
    spec_record = ArchitectureSpecRecord(
        document_id=document.id,
        version=version,
        status=result.spec.status.value,
        spec_json=result.spec.model_dump(mode="json"),
        approved_at=datetime.now(UTC) if auto_generate else None,
    )
    session.add(spec_record)
    session.flush()
    for source in result.official_sources:
        session.add(
            ExternalSource(
                document_id=document.id,
                url=source.url,
                commit_sha=source.commit_sha,
                license=source.license_spdx,
                reference_files=source.reference_files,
                verified_official=source.verified,
            )
        )
    if auto_generate:
        generation = Generation(document_id=document.id, spec_id=spec_record.id)
        session.add(generation)
        session.flush()
        session.add(
            AnalysisRun(
                document_id=document.id,
                generation_id=generation.id,
                kind="generation",
                max_attempts=settings.max_job_attempts,
                event_log=[
                    {
                        "sequence": 1,
                        "stage": "queued",
                        "progress": 0,
                        "message": "automatic code generation queued",
                    }
                ],
            )
        )
        append_event(run, "completed", 100, "analysis complete; code generation queued automatically")
    else:
        append_event(run, "needs_review", 100, "no implementable architecture graph could be extracted")
    run.status = "completed"
    session.commit()


def _custom_modules(spec: ModelGraphSpec, provider: Any) -> dict[str, str]:
    rendered = render_model(spec, {})
    if not rendered.custom_operations:
        return {}
    modules: dict[str, str] = {}
    for node_id in rendered.custom_operations:
        node = next(item for item in spec.nodes if item.id == node_id)
        response = synthesize_custom_module(provider, node)
        node.params["class_name"] = response.class_name
        modules[node.id] = response.source
    return modules


def _request_repair(
    provider: Any,
    spec: ModelGraphSpec,
    result: SandboxResult,
    target: str,
    custom_modules: dict[str, str],
) -> RestrictedRepairPatch:
    patch = provider.generate_structured(
        RestrictedRepairPatch,
        instructions=(
            f"Return one minimal {target} repair for the failed approved model structure. "
            "You may only change a node's params or input order, or replace one standalone custom module. "
            "Do not add imports outside torch/typing/math and do not rewrite the project."
        ),
        prompt=str(
            {
                "failure_category": result.failure_category,
                "failure": result.message,
                "architecture": spec.model_dump(mode="json"),
                "custom_operations": sorted(custom_modules),
            }
        ),
    )
    if patch.target != target:
        raise ValueError(f"repair target mismatch: expected {target}, received {patch.target}")
    node = next((item for item in spec.nodes if item.id == patch.node_id), None)
    if node is None:
        raise ValueError(f"repair references unknown node: {patch.node_id}")
    if patch.params is not None:
        node.params = {**node.params, **patch.params}
    if patch.inputs is not None:
        node.inputs = patch.inputs
    if patch.target in {"custom_block", "glue"} and patch.custom_source:
        if not patch.custom_source or not patch.custom_class_name:
            raise ValueError(f"{patch.target} module repair requires source and class name")
        validate_custom_module(patch.custom_source, patch.custom_class_name)
        node.params["class_name"] = patch.custom_class_name
        custom_modules[node.id] = patch.custom_source
    elif patch.target == "custom_block":
        raise ValueError("custom-block repair requires source and class name")
    ModelGraphSpec.model_validate(spec.model_dump(mode="json"))
    return patch


def process_generation(session: Session, run: AnalysisRun, settings: Settings) -> None:
    generation = session.get(Generation, run.generation_id)
    if generation is None:
        raise ValueError("generation no longer exists")
    spec_record = session.get(ArchitectureSpecRecord, generation.spec_id)
    document = session.get(Document, generation.document_id)
    if spec_record is None or document is None:
        raise ValueError("generation dependencies no longer exist")
    spec = ModelGraphSpec.model_validate(spec_record.spec_json)
    if spec_record.status != "approved":
        raise ValueError("generation requires an approved model structure")

    provider = build_provider(settings)
    append_event(run, "code_generation", 20, "rendering approved model structure")
    session.commit()
    custom_modules = _custom_modules(spec, provider)
    repair_log: list[dict[str, object]] = []
    package = write_package(
        spec,
        settings.storage_root / "artifacts" / generation.id / "attempt-0",
        document_sha256=str(document.sha256),
        spec_version=spec_record.version,
        provider=settings.llm_provider,
        model=provider.identity,
        custom_modules=custom_modules,
        repair_log=repair_log,
    )
    append_event(run, "sandbox", 60, "running isolated compile/forward/backward checks")
    session.commit()
    sandbox = (
        RemoteSandbox(
            settings.sandbox_runner_url,
            settings.storage_root,
            timeout_seconds=settings.sandbox_timeout_seconds,
        )
        if settings.sandbox_runner_url
        else DockerSandbox(
            settings.sandbox_image,
            timeout_seconds=settings.sandbox_timeout_seconds,
            memory=settings.sandbox_memory,
            cpus=settings.sandbox_cpus,
            pids=settings.sandbox_pids_limit,
        )
    )
    result = sandbox.validate(package.path)
    save_validation_result(package.path, result)
    for attempt in range(1, 4):
        if result.status == "passed" or result.failure_category == FailureCategory.SANDBOX:
            break
        target = repair_priority(result.failure_category)[min(attempt - 1, 2)]
        append_event(run, "repair", 60 + attempt * 10, f"attempt {attempt}/3: restricted {target} repair")
        session.commit()
        try:
            patch = _request_repair(provider, spec, result, target, custom_modules)
            repair_log.append({"attempt": attempt, **patch.model_dump(mode="json")})
            package = write_package(
                spec,
                settings.storage_root / "artifacts" / generation.id / f"attempt-{attempt}",
                document_sha256=str(document.sha256),
                spec_version=spec_record.version,
                provider=settings.llm_provider,
                model=provider.identity,
                custom_modules=custom_modules,
                repair_log=repair_log,
            )
            result = sandbox.validate(package.path)
            save_validation_result(package.path, result)
        except Exception as exc:
            repair_log.append({"attempt": attempt, "target": target, "rejected": str(exc)})
        generation.repair_count = attempt
    generation.validation_json = {**result.as_json(), "repair_log": repair_log}
    if result.status != "passed":
        generation.status = "needs_review"
        generation.failure_reason = result.message
        document.status = "needs_review"
        run.status = "completed"
        append_event(run, "needs_review", 100, "sandbox validation failed; no fallback was generated")
        session.commit()
        return

    archive_base = package.path.parent / f"{package.path.name}-artifact"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", package.path))
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    session.add(
        Artifact(
            generation_id=generation.id,
            storage_path=str(archive_path),
            sha256=digest,
            provenance=package.provenance,
        )
    )
    generation.status = "completed"
    document.status = "completed"
    run.status = "completed"
    append_event(run, "completed", 100, "validated model package is ready")
    session.commit()


def process_run(session: Session, run: AnalysisRun, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    try:
        if run.kind == "analysis":
            process_analysis(session, run, settings)
        elif run.kind == "generation":
            process_generation(session, run, settings)
        else:
            raise ValueError(f"unsupported run kind: {run.kind}")
    except Exception as exc:
        session.rollback()
        current_run = session.get(AnalysisRun, run.id)
        if current_run is None:
            raise
        current_run.error = str(exc)
        if current_run.attempts < current_run.max_attempts:
            current_run.status = "queued"
            append_event(current_run, "retrying", current_run.progress, f"retry scheduled: {exc}")
        else:
            current_run.status = "failed"
            append_event(current_run, "failed", 100, str(exc))
            if current_run.generation_id:
                generation = session.get(Generation, current_run.generation_id)
                if generation:
                    generation.status = "needs_review"
                    generation.failure_reason = str(exc)
        session.commit()
