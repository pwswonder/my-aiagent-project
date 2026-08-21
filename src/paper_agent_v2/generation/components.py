from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from paper_agent_v2.ir import NodeSpec


class UnsupportedOperationError(ValueError):
    pass


def _literal(value: Any) -> str:
    return repr(value)


def _module(name: str) -> Callable[[NodeSpec], str]:
    def build(node: NodeSpec) -> str:
        params = ", ".join(f"{key}={_literal(value)}" for key, value in node.params.items())
        return f"nn.{name}({params})"

    return build


@dataclass(frozen=True, slots=True)
class ComponentDefinition:
    constructor: Callable[[NodeSpec], str]
    forward: Callable[[NodeSpec, str], str] = lambda node, args: f"self.{node.id}({args})"


def _concat(node: NodeSpec, args: str) -> str:
    dim = int(node.params.get("dim", 1))
    return f"torch.cat([{args}], dim={dim})"


def _add(node: NodeSpec, args: str) -> str:
    return " + ".join(item.strip() for item in args.split(","))


def _reshape(node: NodeSpec, args: str) -> str:
    return f"{args}.reshape({', '.join(repr(v) for v in node.params['shape'])})"


def _permute(node: NodeSpec, args: str) -> str:
    return f"{args}.permute({', '.join(str(v) for v in node.params['dims'])})"


def _attention(node: NodeSpec, args: str) -> str:
    values = [item.strip() for item in args.split(",")]
    query = values[0]
    key = values[1] if len(values) > 1 else query
    value = values[2] if len(values) > 2 else key
    return f"self.{node.id}({query}, {key}, {value}, need_weights=False)[0]"


def _recurrent(node: NodeSpec, args: str) -> str:
    return f"self.{node.id}({args})[0]"


def _patch_embedding(node: NodeSpec) -> str:
    params = node.params
    return (
        f"PatchEmbedding(in_channels={int(params['in_channels'])}, embed_dim={int(params['embed_dim'])}, "
        f"patch_size={int(params['patch_size'])}, use_norm={params.get('norm') is not None})"
    )


def _pvt_encoder_sra(node: NodeSpec) -> str:
    params = node.params
    depth_value = params.get("num_layers", 1)
    if isinstance(depth_value, dict):
        depth_value = depth_value.get("PVT-Small", next(iter(depth_value.values())))
    return (
        f"PVTEncoderSRA(channels={int(params['embed_dim'])}, num_heads={int(params['num_heads'])}, "
        f"depth={int(depth_value)}, sr_ratio={int(params.get('sr_ratio', 1))}, "
        f"expansion_ratio={int(params.get('ffn_expansion_ratio', 4))}, "
        f"dropout={float(params.get('dropout', 0.0))})"
    )


REGISTRY: dict[str, ComponentDefinition] = {
    "identity": ComponentDefinition(_module("Identity")),
    "linear": ComponentDefinition(_module("Linear")),
    "conv1d": ComponentDefinition(_module("Conv1d")),
    "conv2d": ComponentDefinition(_module("Conv2d")),
    "conv3d": ComponentDefinition(_module("Conv3d")),
    "convtranspose2d": ComponentDefinition(_module("ConvTranspose2d")),
    "batchnorm1d": ComponentDefinition(_module("BatchNorm1d")),
    "batchnorm2d": ComponentDefinition(_module("BatchNorm2d")),
    "layernorm": ComponentDefinition(_module("LayerNorm")),
    "relu": ComponentDefinition(_module("ReLU")),
    "gelu": ComponentDefinition(_module("GELU")),
    "silu": ComponentDefinition(_module("SiLU")),
    "dropout": ComponentDefinition(_module("Dropout")),
    "flatten": ComponentDefinition(_module("Flatten")),
    "maxpool2d": ComponentDefinition(_module("MaxPool2d")),
    "avgpool2d": ComponentDefinition(_module("AvgPool2d")),
    "adaptiveavgpool2d": ComponentDefinition(_module("AdaptiveAvgPool2d")),
    "embedding": ComponentDefinition(_module("Embedding")),
    "multiheadattention": ComponentDefinition(_module("MultiheadAttention"), _attention),
    "transformerencoderlayer": ComponentDefinition(_module("TransformerEncoderLayer")),
    "transformerdecoderlayer": ComponentDefinition(_module("TransformerDecoderLayer")),
    "patchembed": ComponentDefinition(_patch_embedding),
    "transformerencodersra": ComponentDefinition(_pvt_encoder_sra),
    "lstm": ComponentDefinition(_module("LSTM"), _recurrent),
    "gru": ComponentDefinition(_module("GRU"), _recurrent),
    "add": ComponentDefinition(lambda node: "nn.Identity()", _add),
    "residual": ComponentDefinition(lambda node: "nn.Identity()", _add),
    "skipconnection": ComponentDefinition(lambda node: "nn.Identity()", _add),
    "concat": ComponentDefinition(lambda node: "nn.Identity()", _concat),
    "unetskip": ComponentDefinition(lambda node: "nn.Identity()", _concat),
    "reshape": ComponentDefinition(lambda node: "nn.Identity()", _reshape),
    "permute": ComponentDefinition(lambda node: "nn.Identity()", _permute),
    "graphconv": ComponentDefinition(
        lambda node: (
            f"GraphConv(in_features={int(node.params['in_features'])}, out_features={int(node.params['out_features'])})"
        )
    ),
    "residualblock": ComponentDefinition(
        lambda node: (
            f"ResidualBlock(channels={int(node.params['channels'])}, "
            f"kernel_size={int(node.params.get('kernel_size', 3))})"
        )
    ),
    "vaereparameterization": ComponentDefinition(lambda node: "VAEReparameterization()"),
    "timeseriesdecomposition": ComponentDefinition(
        lambda node: f"MovingAverageDecomposition(kernel_size={int(node.params.get('kernel_size', 25))})"
    ),
    "multimodalfusion": ComponentDefinition(
        lambda node: (
            f"MultimodalFusion(input_dim={int(node.params['input_dim'])}, output_dim={int(node.params['output_dim'])})"
        )
    ),
}


def normalize_op(op: str) -> str:
    return "".join(character for character in op.lower() if character.isalnum())


def definition_for(node: NodeSpec) -> ComponentDefinition:
    try:
        return REGISTRY[normalize_op(node.op)]
    except KeyError as exc:
        raise UnsupportedOperationError(f"unsupported operation: {node.op}") from exc
