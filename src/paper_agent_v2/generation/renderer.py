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


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, use_norm: bool = True) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        height, width = x.shape[-2:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x.transpose(1, 2).reshape(x.shape[0], -1, height, width)


class SpatialReductionEncoderBlock(nn.Module):
    def __init__(
        self, channels: int, num_heads: int, sr_ratio: int, expansion_ratio: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.sr_ratio = sr_ratio
        self.norm1 = nn.LayerNorm(channels)
        self.reduction = (
            nn.Conv2d(channels, channels, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        )
        self.reduction_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = channels * expansion_ratio
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        query = self.norm1(tokens)
        reduced = self.reduction(x).flatten(2).transpose(1, 2)
        key_value = self.reduction_norm(reduced)
        attended, _ = self.attention(query, key_value, key_value, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.feed_forward(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class PVTEncoderSRA(nn.Module):
    def __init__(
        self, channels: int, num_heads: int, depth: int, sr_ratio: int, expansion_ratio: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SpatialReductionEncoderBlock(channels, num_heads, sr_ratio, expansion_ratio, dropout)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class RepeatPadLastObservation(nn.Module):
    def __init__(self, pad_steps: int = 4) -> None:
        super().__init__()
        self.pad_steps = pad_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, x[..., -1:].expand(*x.shape[:-1], self.pad_steps)], dim=-1)


class SlidingWindowPatchify(nn.Module):
    def __init__(self, patch_length: int = 8, stride: int = 4) -> None:
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < self.patch_length:
            x = torch.nn.functional.pad(x, (0, self.patch_length - x.shape[-1]), mode="replicate")
        return x.unfold(dimension=-1, size=self.patch_length, step=self.stride)


class FixedPositionalEncoding(nn.Module):
    def __init__(self, model_dim: int = 32) -> None:
        super().__init__()
        self.model_dim = model_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.model_dim, 2, device=x.device, dtype=x.dtype)
            * (-torch.log(torch.tensor(10000.0, device=x.device, dtype=x.dtype)) / self.model_dim)
        )
        encoding = torch.zeros(x.shape[-2], self.model_dim, device=x.device, dtype=x.dtype)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        return x + encoding


class TimeSeriesTransformerEncoder(nn.Module):
    def __init__(
        self,
        model_dim: int = 32,
        num_heads: int = 4,
        ff_dim: int = 64,
        layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-2]
        encoded = self.encoder(x.reshape(-1, x.shape[-2], x.shape[-1]))
        return encoded.reshape(*leading_shape, x.shape[-2], x.shape[-1])


class PerChannelLinear(nn.Module):
    def __init__(self, in_features: int = 32, out_features: int = 8, channels: int = 4) -> None:
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(channels)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 4 or x.shape[1] != len(self.heads):
            return self.heads[0](x)
        return torch.stack([head(x[:, index]) for index, head in enumerate(self.heads)], dim=1)


class LastPatchPair(nn.Module):
    def forward(
        self, expected: torch.Tensor, reconstructed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return expected, reconstructed


class SquaredErrorReduce(nn.Module):
    def forward(
        self,
        expected: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        reconstructed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if reconstructed is None:
            expected, reconstructed = expected
        error = (expected[..., -1, :] - reconstructed[..., -1, :]).square()
        return error.reshape(error.shape[0], -1).sum(dim=1)


class PatchwiseL2Error(nn.Module):
    def forward(self, expected: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        error = (expected - reconstructed).square()
        return error.reshape(error.shape[0], -1).sum(dim=1)


class WindowReconstructionError(nn.Module):
    def forward(self, expected: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        if expected.dim() + 1 == reconstructed.dim():
            patch_length = reconstructed.shape[-1]
            stride = 6
            expected = torch.cat(
                [expected, expected[..., -1:].expand(*expected.shape[:-1], stride)], dim=-1
            ).unfold(-1, patch_length, stride)
        patch_count = min(expected.shape[-2], reconstructed.shape[-2])
        error = (expected[..., -patch_count:, :] - reconstructed[..., -patch_count:, :]).square()
        return error.reshape(error.shape[0], -1).sum(dim=1)
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
        raise ValueError(f"unsafe Python identifier in model structure: {value!r}")


def render_model(spec: ModelGraphSpec, custom_modules: dict[str, str] | None = None) -> RenderedModel:
    """Render an approved DAG. Jinja is intentionally not involved in model code."""
    if spec.status != SpecStatus.APPROVED:
        raise ValueError("model structure must be approved before generation")
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
        try:
            definition = definition_for(node)
        except UnsupportedOperationError:
            source = custom_modules.get(node.id) or custom_modules.get(node.op)
            if source is None:
                custom_operations.append(node.id)
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
