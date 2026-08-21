from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from paper_agent_v2.generation.custom import CustomModuleResponse, synthesize_custom_module, validate_custom_module
from paper_agent_v2.generation.package_writer import _shape_literal, write_package
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


def test_repeated_custom_operation_uses_a_module_per_node() -> None:
    spec = ModelGraphSpec(
        name="Custom stages",
        task="features",
        inputs=[TensorSpec(name="x", shape=["B", 4])],
        nodes=[
            NodeSpec(
                id="stage1",
                op="novel_projection",
                inputs=["x"],
                output="hidden",
                params={"class_name": "StageOne"},
                evidence_ids=["e1"],
            ),
            NodeSpec(
                id="stage2",
                op="novel_projection",
                inputs=["hidden"],
                output="output",
                params={"class_name": "StageTwo"},
                evidence_ids=["e1"],
            ),
        ],
        outputs=[TensorSpec(name="output", shape=["B", 4])],
        evidence=[EvidenceRef(id="e1", source_type=EvidenceSource.PDF, page=2, quote="Two stages.")],
    ).approve()
    missing = render_model(spec)
    assert missing.custom_operations == ["stage1", "stage2"]

    rendered = render_model(
        spec,
        {
            "stage1": "class StageOne(nn.Module):\n    def forward(self, x): return x",
            "stage2": "class StageTwo(nn.Module):\n    def forward(self, x): return x",
        },
    )

    assert "self.stage1 = StageOne()" in rendered.source
    assert "self.stage2 = StageTwo()" in rendered.source
    ast.parse(rendered.source)


def test_condition_metadata_does_not_block_custom_module_synthesis() -> None:
    spec = approved_spec()
    spec.nodes[0].op = "semantic_clustering"
    spec.nodes[0].condition = "when class labels are unavailable"

    rendered = render_model(spec)

    assert rendered.custom_operations == [spec.nodes[0].id]

def test_pvt_components_are_rendered_from_registry() -> None:
    spec = ModelGraphSpec(
        name="PVT test",
        task="features",
        inputs=[TensorSpec(name="image", shape=["B", 3, "H", "W"])],
        nodes=[
            NodeSpec(
                id="patch",
                op="patch_embed",
                inputs=["image"],
                output="features",
                params={"in_channels": 3, "embed_dim": 32, "patch_size": 4, "norm": "LayerNorm"},
                evidence_ids=["e1"],
            ),
            NodeSpec(
                id="encoder",
                op="transformer_encoder_sra",
                inputs=["features"],
                output="output",
                params={
                    "embed_dim": 32,
                    "num_heads": 1,
                    "num_layers": {"PVT-Small": 1},
                    "sr_ratio": 2,
                    "ffn_expansion_ratio": 4,
                },
                evidence_ids=["e1"],
            ),
        ],
        outputs=[TensorSpec(name="output", shape=["B", 32, "H/4", "W/4"])],
        evidence=[EvidenceRef(id="e1", source_type=EvidenceSource.PDF, page=4, quote="PVT stage.")],
    ).approve()

    rendered = render_model(spec)

    assert rendered.custom_operations == []
    assert "PatchEmbedding(" in rendered.source
    assert "PVTEncoderSRA(" in rendered.source
    ast.parse(rendered.source)


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


def test_symbolic_image_inputs_use_forward_safe_spatial_defaults() -> None:
    assert _shape_literal(["B", 3, "H", "W"]) == "[2, 3, 224, 224]"


def test_integer_example_inputs_do_not_use_randn(tmp_path: Path) -> None:
    spec = approved_spec()
    spec.inputs.append(TensorSpec(name="class_label", shape=["B"], dtype="int64"))
    package = write_package(
        spec,
        tmp_path,
        document_sha256="c" * 64,
        spec_version=1,
        provider="test",
        model="static:test",
    )

    source = (package.path / "example_inputs.py").read_text()

    assert '"class_label": torch.zeros(*[2], dtype=torch.int64)' in source


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
        "import torch\nfrom torch import nn\nclass Novel(nn.Module):\n"
        "    def __init__(self): super().__init__()\n"
        "    def forward(self, x): return x",
        "Novel",
    )
    with pytest.raises(ValueError, match="forbidden"):
        validate_custom_module("import subprocess\nfrom torch import nn\nclass Novel(nn.Module):\n    pass", "Novel")
    with pytest.raises(ValueError, match="forbidden call"):
        validate_custom_module(
            "from torch import nn\nclass Novel(nn.Module):\n    def forward(self, x): open('/tmp/x', 'w')",
            "Novel",
        )
    with pytest.raises(ValueError, match="dunder attribute"):
        validate_custom_module(
            "from torch import nn\nclass Novel(nn.Module):\n"
            "    def forward(self, x): return x.__class__",
            "Novel",
        )


def test_custom_module_generation_retries_invalid_class_contract() -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def generate_structured(self, schema, *, instructions, prompt):
            self.calls += 1
            class_name = "Expected" if self.calls == 1 else "Actual"
            return CustomModuleResponse(
                class_name=class_name,
                source="from torch import nn\nclass Actual(nn.Module):\n    def forward(self, x): return x",
            )

    provider = Provider()
    node = NodeSpec(id="novel", op="novel", inputs=["x"], output="y", evidence_ids=[])
    response = synthesize_custom_module(provider, node)  # type: ignore[arg-type]

    assert response.class_name == "Actual"
    assert provider.calls == 2
