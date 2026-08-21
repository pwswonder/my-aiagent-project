from __future__ import annotations

from collections import defaultdict, deque
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SpecStatus(StrEnum):
    DRAFT = "draft"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"


class EvidenceSource(StrEnum):
    PDF = "pdf"
    OFFICIAL_CODE = "official_code"


class EvidenceRef(BaseModel):
    id: str
    source_type: EvidenceSource
    quote: str = Field(min_length=1, max_length=2_000)
    page: int | None = Field(default=None, ge=1)
    section: str | None = None
    url: str | None = None
    commit_sha: str | None = None
    chunk_id: str | None = None


class TensorSpec(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    shape: list[int | str] = Field(min_length=1)
    dtype: str = "float32"
    semantics: str | None = None

    @field_validator("shape")
    @classmethod
    def validate_symbolic_shape(cls, shape: list[int | str]) -> list[int | str]:
        import re

        for dimension in shape:
            if isinstance(dimension, int) and dimension <= 0:
                raise ValueError("concrete tensor dimensions must be positive")
            if isinstance(dimension, str) and not (
                dimension == "..." or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:[*/+-]\d+)*", dimension)
            ):
                raise ValueError(f"invalid symbolic tensor dimension: {dimension!r}")
        return shape


class NodeSpec(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    op: str
    inputs: list[str] = Field(min_length=1)
    output: str
    params: dict[str, Any] = Field(default_factory=dict)
    repeats: int = Field(default=1, ge=1, le=1_000)
    condition: str | None = None
    share_with: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class TrainingSpec(BaseModel):
    loss: str | None = None
    optimizer: str | None = None
    scheduler: str | None = None
    initialization: str | None = None
    metrics: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class Assumption(BaseModel):
    field: str
    value: Any
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class UnresolvedItem(BaseModel):
    field: str
    question: str
    blocking: bool = True
    evidence_ids: list[str] = Field(default_factory=list)


class ModelGraphSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    name: str
    task: str
    framework: Literal["pytorch"] = "pytorch"
    status: SpecStatus = SpecStatus.DRAFT
    inputs: list[TensorSpec]
    nodes: list[NodeSpec]
    outputs: list[TensorSpec]
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    unresolved: list[UnresolvedItem] = Field(default_factory=list)
    parameter_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_graph_contract(self) -> ModelGraphSpec:
        evidence_ids = {item.id for item in self.evidence}
        if len(evidence_ids) != len(self.evidence):
            raise ValueError("evidence ids must be unique")
        if len({node.id for node in self.nodes}) != len(self.nodes):
            raise ValueError("node ids must be unique")
        if len({node.output for node in self.nodes}) != len(self.nodes):
            raise ValueError("node output tensors must be unique")
        node_ids = {node.id for node in self.nodes}
        for node in self.nodes:
            if node.share_with and node.share_with not in node_ids:
                raise ValueError(f"node {node.id} shares weights with an unknown node")

        produced = {item.name for item in self.inputs}
        for node in self.topological_nodes():
            missing_inputs = set(node.inputs) - produced
            if missing_inputs:
                raise ValueError(f"node {node.id} references unavailable tensors: {sorted(missing_inputs)}")
            produced.add(node.output)
            unknown_evidence = set(node.evidence_ids) - evidence_ids
            if unknown_evidence:
                raise ValueError(f"node {node.id} references unknown evidence: {sorted(unknown_evidence)}")
            has_assumption = any(a.field.startswith(f"nodes.{node.id}") for a in self.assumptions)
            if not node.evidence_ids and not has_assumption:
                raise ValueError(f"node {node.id} requires evidence or an explicit assumption")

        if set(self.training.evidence_ids) - evidence_ids:
            raise ValueError("training references unknown evidence")
        missing_outputs = {item.name for item in self.outputs} - produced
        if missing_outputs:
            raise ValueError(f"outputs are not produced by the graph: {sorted(missing_outputs)}")
        if any(item.blocking for item in self.unresolved):
            self.status = SpecStatus.NEEDS_REVIEW
        return self

    def topological_nodes(self) -> list[NodeSpec]:
        producers = {node.output: node.id for node in self.nodes}
        by_id = {node.id: node for node in self.nodes}
        indegree = {node.id: 0 for node in self.nodes}
        children: dict[str, list[str]] = defaultdict(list)
        for node in self.nodes:
            deps = {producers[name] for name in node.inputs if name in producers}
            if node.id in deps:
                raise ValueError(f"node {node.id} consumes its own output")
            indegree[node.id] = len(deps)
            for dep in deps:
                children[dep].append(node.id)

        ready = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
        ordered: list[NodeSpec] = []
        while ready:
            node_id = ready.popleft()
            ordered.append(by_id[node_id])
            for child in sorted(children[node_id]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(ordered) != len(self.nodes):
            raise ValueError("architecture graph contains a cycle")
        return ordered

    def approve(self) -> ModelGraphSpec:
        if any(item.blocking for item in self.unresolved):
            raise ValueError("blocking unresolved items must be resolved before approval")
        self.status = SpecStatus.APPROVED
        return self
