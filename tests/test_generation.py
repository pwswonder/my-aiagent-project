from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from paper_agent_v2.generation.custom import validate_custom_module
from paper_agent_v2.generation.package_writer import write_package
from paper_agent_v2.generation.renderer import render_model
from paper_agent_v2.ir import EvidenceRef, EvidenceSource, ModelGraphSpec, NodeSpec, TensorSpec


def approved_spec() -> ModelGraphSpec:
    spec = ModelGraphSpec(
        name="Tiny/Net",
        task="classification",
        inputs=[TensorSpec(name="x", shape=["B", 4])],
        nodes=[
            NodeSpec(
                id="projection",
                op="Linear",
                inputs=["x"],
                output="hidden",
                params={"in_features": 4, "out_features": 4},
                evidence_ids=["e1"],
            ),
            NodeSpec(
                id="residual",
                op="Add",
                inputs=["hidden", "x"],
                output="output",
                evidence_ids=["e1"],
            ),
        ],
        outputs=[TensorSpec(name="output", shape=["B", 4])],
        evidence=[EvidenceRef(id="e1", source_type=EvidenceSource.PDF, page=2, quote="Residual projection.")],
    )
    return spec.approve()


def test_renderer_is_deterministic_and_not_jinja_model_code() -> None:
    first = render_model(approved_spec()).source
    second = render_model(approved_spec()).source
    assert first == second
    assert "hidden + x" in first
    ast.parse(first)


def test_package_contains_provenance_and_scaffolding(tmp_path: Path) -> None:
    package = write_package(
        approved_spec(),
        tmp_path,
        document_sha256="a" * 64,
        spec_version=3,
        provider="test",
        model="static:test",
    )
    expected = {
        "model.py",
        "config.py",
        "example_inputs.py",
        "README.md",
        "architecture.json",
        "provenance.json",
        "validation.json",
        "tests/test_model.py",
    }
    assert expected == {str(path.relative_to(package.path)) for path in package.path.rglob("*") if path.is_file()}
    assert json.loads((package.path / "provenance.json").read_text())["spec_version"] == 3


def test_generated_model_runs_forward_backward(tmp_path: Path) -> None:
    import torch

    package = write_package(
        approved_spec(),
        tmp_path,
        document_sha256="b" * 64,
        spec_version=1,
        provider="test",
        model="static:test",
    )
    module_spec = importlib.util.spec_from_file_location("generated_model", package.path / "model.py")
    assert module_spec and module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = module.PaperModel()
    output = model(torch.randn(2, 4))
    assert tuple(output.shape) == (2, 4)
    output.mean().backward()
    torch.optim.Adam(model.parameters(), lr=1e-4).step()


def test_custom_module_security_contract() -> None:
    validate_custom_module(
        "import torch\nfrom torch import nn\nclass Novel(nn.Module):\n    def forward(self, x): return x",
        "Novel",
    )
    with pytest.raises(ValueError, match="forbidden"):
        validate_custom_module("import subprocess\nfrom torch import nn\nclass Novel(nn.Module):\n    pass", "Novel")
    with pytest.raises(ValueError, match="forbidden call"):
        validate_custom_module(
            "from torch import nn\nclass Novel(nn.Module):\n    def forward(self, x): open('/tmp/x', 'w')",
            "Novel",
        )
