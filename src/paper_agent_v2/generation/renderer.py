from __future__ import annotations

import keyword
from dataclasses import dataclass

from paper_agent_v2.generation.components import UnsupportedOperationError, definition_for
from paper_agent_v2.ir import ModelGraphSpec, SpecStatus

HELPERS = """
class GraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.matmul(adjacency, x))


class MovingAverageDecomposition(nn.Module):
    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.pool(x.transpose(1, 2)).transpose(1, 2)
        return x - trend, trend


class MultimodalFusion(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.cat(inputs, dim=-1))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class VAEReparameterization(nn.Module):
    def forward(self, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        standard_deviation = torch.exp(0.5 * log_variance)
        return mean + torch.randn_like(standard_deviation) * standard_deviation
"""


@dataclass(slots=True)
class RenderedModel:
    source: str
    custom_operations: list[str]


def _custom_forward(node_id: str):
    def build(args: str) -> str:
        return f"self.{node_id}({args})"

    return build


def _registry_forward(definition, node):
    def build(args: str) -> str:
        return definition.forward(node, args)

    return build


def _assert_identifier(value: str) -> None:
    if not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError(f"unsafe Python identifier in Architecture IR: {value!r}")


def render_model(spec: ModelGraphSpec, custom_modules: dict[str, str] | None = None) -> RenderedModel:
    """Render an approved DAG. Jinja is intentionally not involved in model code."""
    if spec.status != SpecStatus.APPROVED:
        raise ValueError("Architecture IR must be approved before generation")
    if any(item.blocking for item in spec.unresolved):
        raise ValueError("blocking unresolved items remain")

    custom_modules = custom_modules or {}
    custom_sources: list[str] = []
    custom_operations: list[str] = []
    init_lines: list[str] = []
    forward_lines: list[str] = []

    for tensor in [*spec.inputs, *spec.outputs]:
        _assert_identifier(tensor.name)

    for node in spec.topological_nodes():
        _assert_identifier(node.id)
        _assert_identifier(node.output)
        for item in node.inputs:
            _assert_identifier(item)
        if node.condition:
            raise ValueError(f"conditional node {node.id!r} requires an explicit custom module")

        try:
            definition = definition_for(node)
        except UnsupportedOperationError:
            source = custom_modules.get(node.op)
            if source is None:
                custom_operations.append(node.op)
                continue
            if source not in custom_sources:
                custom_sources.append(source.strip())
            class_name = str(node.params.get("class_name", node.op))
            _assert_identifier(class_name)
            constructor_args = str(node.params.get("constructor_args", ""))
            constructor = f"{class_name}({constructor_args})"

            forward_builder = _custom_forward(node.id)

        else:
            constructor = definition.constructor(node)

            forward_builder = _registry_forward(definition, node)

        forward_expression = forward_builder(", ".join(node.inputs))

        if node.share_with:
            _assert_identifier(node.share_with)
            init_lines.append(f"        self.{node.id} = self.{node.share_with}")
        elif node.repeats > 1:
            init_lines.append(f"        self.{node.id} = nn.ModuleList([{constructor} for _ in range({node.repeats})])")
        else:
            init_lines.append(f"        self.{node.id} = {constructor}")

        if node.repeats > 1:
            current = node.inputs[0]
            extra = node.inputs[1:]
            forward_lines.append(f"        {node.output} = {current}")
            if node.share_with:
                forward_lines.append(f"        for _ in range({node.repeats}):")
            else:
                forward_lines.append(f"        for layer in self.{node.id}:")
            repeated_args = ", ".join([node.output, *extra])
            repeated = forward_builder(repeated_args)
            if not node.share_with:
                repeated = repeated.replace(f"self.{node.id}", "layer", 1)
            forward_lines.append(f"            {node.output} = {repeated}")
        else:
            forward_lines.append(f"        {node.output} = {forward_expression}")

    if custom_operations:
        return RenderedModel(source="", custom_operations=sorted(set(custom_operations)))

    arguments = ", ".join(f"{item.name}: torch.Tensor" for item in spec.inputs)
    outputs = [item.name for item in spec.outputs]
    return_value = outputs[0] if len(outputs) == 1 else f"({', '.join(outputs)})"
    custom_text = "\n\n".join(custom_sources)
    source = f"""from __future__ import annotations

import torch
from torch import nn

{HELPERS}
{custom_text}

class PaperModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
{chr(10).join(init_lines) if init_lines else "        pass"}

    def forward(self, {arguments}):
{chr(10).join(forward_lines)}
        return {return_value}
"""
    return RenderedModel(source=source, custom_operations=[])
