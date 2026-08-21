from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from paper_agent_v2.ir import (
    ModelGraphSpec,
    SpecStatus,
    TensorSpec,
    UnresolvedItem,
)
from paper_agent_v2.official_code import GitHubSourceResolver, OfficialCodeSource
from paper_agent_v2.parser import ParsedPaper, parse_pdf, render_pages
from paper_agent_v2.providers.base import LLMProvider
from paper_agent_v2.retrieval import HybridRetriever

ARCHITECTURE_QUERIES = (
    "proposed model architecture layers blocks inputs outputs skip connections",
    "implementation details hidden dimensions heads depth kernel stride",
    "training objective loss optimizer initialization",
)

SYSTEM_PROMPT = """You extract a faithful PyTorch architecture graph from research evidence.
Return only the supplied JSON schema. Every node and training hyperparameter must cite evidence_ids.
If evidence is missing, add an assumption with a confidence and an unresolved item when it can affect
model behavior. Do not replace an unknown architecture with a generic CNN, MLP, or Transformer.
Use symbolic dimensions such as B, C, H, W, T, D. Node inputs reference input tensor names or earlier
node outputs. share_with may only reference the id of another node in the returned nodes array; omit it
unless the paper explicitly shares weights. Mark the spec needs_review whenever a blocking choice remains."""


@dataclass(slots=True)
class AnalysisResult:
    paper: ParsedPaper
    spec: ModelGraphSpec
    summary: str
    page_analysis: str | None
    official_sources: list[OfficialCodeSource]


def _evidence_payload(paper: ParsedPaper) -> list[dict[str, object]]:
    retriever = HybridRetriever(paper.chunks)
    selected = {}
    for query in ARCHITECTURE_QUERIES:
        for hit in retriever.search(query, limit=12):
            selected[hit.chunk.id] = hit.chunk
    # PDF layout blocks are frequently only headings or split fragments. Pull
    # their local neighbours into the extraction context so a heading such as
    # "Proposed Method" includes the actual method paragraphs that follow it.
    ordered = sorted(paper.chunks, key=lambda item: (item.page, item.id))
    positions = {chunk.id: index for index, chunk in enumerate(ordered)}
    for chunk_id in list(selected):
        position = positions[chunk_id]
        for neighbour in ordered[max(0, position - 2) : position + 4]:
            if neighbour.section != "references":
                selected[neighbour.id] = neighbour
    return [
        {
            "id": chunk.id,
            "page": chunk.page,
            "section": chunk.section,
            "kind": chunk.kind,
            "text": chunk.text,
        }
        for chunk in sorted(selected.values(), key=lambda item: (item.page, item.id))
    ]


def needs_review_spec(title: str, reason: str) -> ModelGraphSpec:
    return ModelGraphSpec(
        name=title,
        task="unknown",
        status=SpecStatus.NEEDS_REVIEW,
        inputs=[TensorSpec(name="input", shape=["B", "..."])],
        nodes=[],
        outputs=[TensorSpec(name="input", shape=["B", "..."])],
        unresolved=[UnresolvedItem(field="architecture", question=reason, blocking=True)],
    )


def analyze_paper(
    pdf_path: Path,
    provider: LLMProvider,
    *,
    max_bytes: int,
    title_hint: str | None = None,
    render_dir: Path | None = None,
    source_resolver: GitHubSourceResolver | None = None,
) -> AnalysisResult:
    paper = parse_pdf(pdf_path, max_bytes=max_bytes)
    if title_hint and paper.title.strip().lower() in {"source", "untitled", "document"}:
        paper.title = title_hint.strip()
    official_sources: list[OfficialCodeSource] = []
    if source_resolver:
        for url in paper.github_urls:
            try:
                source = source_resolver.resolve(url, paper.title)
            except Exception:
                source = None
            if source and source.verified:
                official_sources.append(source)
    page_analysis = None
    if render_dir and paper.architecture_pages:
        images = render_pages(pdf_path, paper.architecture_pages, render_dir)
        if images:
            page_analysis = provider.analyze_images(
                images,
                instructions=(
                    "Describe only architecture facts visible in figures, tables, equations and captions. "
                    "State page numbers and explicitly label uncertainty."
                ),
                prompt=f"Paper title: {paper.title}. Extract implementation-relevant visual evidence.",
            )

    evidence = _evidence_payload(paper)
    prompt = json.dumps(
        {
            "title": paper.title,
            "paper_sha256": paper.sha256,
            "text_evidence": evidence,
            "visual_evidence": page_analysis,
            "candidate_official_repositories": paper.github_urls,
            "verified_official_code": [
                {
                    "url": source.url,
                    "commit_sha": source.commit_sha,
                    "license": source.license_spdx,
                    "reference_excerpts": source.reference_excerpts,
                }
                for source in official_sources
            ],
        },
        ensure_ascii=False,
    )
    try:
        spec = provider.generate_structured(
            ModelGraphSpec,
            instructions=SYSTEM_PROMPT,
            prompt=prompt,
        )
    except Exception as exc:
        spec = needs_review_spec(paper.title, f"Architecture extraction failed: {exc}")
    if not spec.nodes and not spec.unresolved:
        spec.unresolved.append(
            UnresolvedItem(
                field="architecture",
                question=(
                    "No implementable architecture nodes were extracted. "
                    "Review the paper evidence and define the model graph."
                ),
                blocking=True,
            )
        )
        spec.status = SpecStatus.NEEDS_REVIEW
    if spec.status == SpecStatus.APPROVED:
        spec.status = SpecStatus.DRAFT
    summary_sections = ("abstract", "introduction", "method", "experiments", "results", "conclusion")
    summary_chunks = []
    for section in summary_sections:
        summary_chunks.extend(chunk for chunk in paper.chunks if chunk.section == section)
    if not summary_chunks:
        summary_chunks = paper.chunks
    summary_evidence = [
        {
            "page": chunk.page,
            "section": chunk.section,
            "text": chunk.text,
        }
        for chunk in summary_chunks[:24]
    ]
    try:
        summary = provider.generate_text(
            instructions=(
                "Summarize the supplied research paper evidence in Korean. Include the research problem, "
                "proposed method, main contributions, training/evaluation setup, and limitations. "
                "Do not invent facts that are absent from the evidence."
            ),
            prompt=json.dumps({"title": paper.title, "evidence": summary_evidence}, ensure_ascii=False),
        )
    except Exception:
        summary = "\n\n".join(chunk.text for chunk in paper.chunks[:3])[:4_000]
    return AnalysisResult(
        paper=paper,
        spec=spec,
        summary=summary,
        page_analysis=page_analysis,
        official_sources=official_sources,
    )
