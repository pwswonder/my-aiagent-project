from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from paper_agent_v2.ir import ModelGraphSpec


class DocumentAccepted(BaseModel):
    document_id: str
    analysis_run_id: str
    status: str


class RunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    document_id: str | None
    generation_id: str | None
    kind: str
    status: str
    stage: str
    progress: int
    attempts: int
    error: str | None
    event_log: list[dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class SpecResponse(BaseModel):
    document_id: str
    version: int
    status: str
    spec: ModelGraphSpec


class ApproveSpecRequest(BaseModel):
    accept_blocking_as_assumptions: bool = False


class GenerationAccepted(BaseModel):
    generation_id: str
    run_id: str
    status: str


class GenerationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    document_id: str
    spec_id: str
    status: str
    repair_count: int
    failure_reason: str | None
    validation_json: dict[str, Any] | None
    artifact_id: str | None = None


class DocumentHistoryItem(BaseModel):
    id: str
    filename: str
    title: str | None
    status: str
    created_at: datetime
    analysis_run_id: str | None = None
    analysis_status: str | None = None
    analysis_stage: str | None = None
    analysis_progress: int = 0


class QATurnResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    question: str
    answer: str | None
    answerability: str
    citations: list[dict[str, Any]]
    created_at: datetime


class DocumentWorkspace(BaseModel):
    document_id: str
    filename: str
    title: str | None
    status: str
    created_at: datetime
    summary: str | None
    analysis_run: RunResponse | None = None
    spec: SpecResponse | None = None
    generation: GenerationResponse | None = None
    generation_run: RunResponse | None = None
    qa_history: list[QATurnResponse] = Field(default_factory=list)


class ArtifactPreview(BaseModel):
    files: dict[str, str]


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4_000)
    limit: int = Field(default=6, ge=1, le=20)


class Citation(BaseModel):
    page: int
    section: str
    chunk_id: str
    evidence: str


class QAResponse(BaseModel):
    answer: str | None
    citations: list[Citation]
    answerability: str
    retrieval_debug: dict[str, Any]
