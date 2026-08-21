from __future__ import annotations

import pytest
from pydantic import ValidationError

from paper_agent_v2.ir import (
    Assumption,
    EvidenceRef,
    EvidenceSource,
    ModelGraphSpec,
    NodeSpec,
    TensorSpec,
    UnresolvedItem,
)


def make_spec(**changes) -> ModelGraphSpec:
    payload = {
        "name": "TinyNet",
        "task": "classification",
        "inputs": [TensorSpec(name="x", shape=["B", 4])],
        "nodes": [
            NodeSpec(
                id="projection",
                op="Linear",
                inputs=["x"],
                output="logits",
                params={"in_features": 4, "out_features": 2},
                evidence_ids=["e1"],
            )
        ],
        "outputs": [TensorSpec(name="logits", shape=["B", 2])],
        "evidence": [
            EvidenceRef(id="e1", source_type=EvidenceSource.PDF, page=3, quote="A linear classifier is used.")
        ],
    }
    payload.update(changes)
    return ModelGraphSpec(**payload)


def test_approved_spec_requires_user_resolution() -> None:
    spec = make_spec(unresolved=[UnresolvedItem(field="nodes.projection.bias", question="Use bias?")])
    assert spec.status == "needs_review"
    with pytest.raises(ValueError, match="blocking"):
        spec.approve()


def test_node_requires_evidence_or_explicit_assumption() -> None:
    with pytest.raises(ValidationError, match="requires evidence"):
        make_spec(nodes=[NodeSpec(id="projection", op="Linear", inputs=["x"], output="logits")])
    spec = make_spec(
        nodes=[NodeSpec(id="projection", op="Linear", inputs=["x"], output="logits")],
        assumptions=[
            Assumption(field="nodes.projection.params", value={}, reason="Paper omits parameters", confidence=0.2)
        ],
    )
    assert spec.nodes[0].id == "projection"


def test_cycle_is_rejected() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        make_spec(
            nodes=[
                NodeSpec(id="a", op="Add", inputs=["b_out"], output="a_out", evidence_ids=["e1"]),
                NodeSpec(id="b", op="Add", inputs=["a_out"], output="b_out", evidence_ids=["e1"]),
            ],
            outputs=[TensorSpec(name="b_out", shape=["B", 2])],
        )
