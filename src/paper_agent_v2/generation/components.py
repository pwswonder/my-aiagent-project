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


SYMBOLIC_INT_DEFAULTS = {
    "P_len": 8,
    "Plen": 8,
    "S": 6,
    "D_model": 32,
    "Dmodel": 32,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 64,
    "M": 4,
}


def _int_value(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return SYMBOLIC_INT_DEFAULTS.get(value, default)
    return default


def _float_value(value: Any, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _linear(node: NodeSpec) -> str:
    params = node.params
    weight_shape = params.get("weight_shape", [])
    if isinstance(weight_shape, list) and len(weight_shape) >= 2:
        in_features = weight_shape[-2]
        out_features = weight_shape[-1]
    else:
        in_features = params.get("in_features")
        out_features = params.get("out_features")
    bias = params.get("bias", True)
    if not isinstance(bias, bool):
        bias = True
    return (
        f"nn.Linear(in_features={_int_value(in_features, 8)}, "
        f"out_features={_int_value(out_features, 32)}, bias={bias})"
    )


def _repeat_pad(node: NodeSpec) -> str:
    return f"RepeatPadLastObservation(pad_steps={_int_value(node.params.get('pad_steps'), 4)})"


def _patchify(node: NodeSpec) -> str:
    patch_length = node.params.get("patch_length", node.params.get("patch_len"))
    return (
        f"SlidingWindowPatchify(patch_length={_int_value(patch_length, 8)}, "
        f"stride={_int_value(node.params.get('stride'), 4)})"
    )


def _pad_and_extract_patches(node: NodeSpec) -> str:
    params = node.params
    patch_length = _int_value(params.get("patch_len", params.get("patch_length")), 8)
    stride = _int_value(params.get("stride"), 6)
    return (
        "nn.Sequential("
        f"RepeatPadLastObservation(pad_steps={stride}), "
        f"SlidingWindowPatchify(patch_length={patch_length}, stride={stride})"
        ")"
    )


def _fixed_position(node: NodeSpec) -> str:
    encoding_shape = node.params.get("encoding_shape", node.params.get("shape", []))
    model_dim = encoding_shape[-1] if isinstance(encoding_shape, list) and encoding_shape else None
    return f"FixedPositionalEncoding(model_dim={_int_value(model_dim, 32)})"


def _linear_add_fixed_position(node: NodeSpec) -> str:
    projection_shape = node.params.get("w_proj_shape", ["Plen", "Dmodel"])
    if not isinstance(projection_shape, list) or len(projection_shape) < 2:
        projection_shape = ["Plen", "Dmodel"]
    in_features = _int_value(projection_shape[-2], 8)
    out_features = _int_value(node.params.get("d_model", projection_shape[-1]), 32)
    return (
        "nn.Sequential("
        f"nn.Linear(in_features={in_features}, out_features={out_features}, bias=False), "
        f"FixedPositionalEncoding(model_dim={out_features})"
        ")"
    )


def _time_series_transformer(node: NodeSpec) -> str:
    params = node.params
    dropout = 0.1 if params.get("dropout") == "present" else _float_value(params.get("dropout"), 0.0)
    return (
        f"TimeSeriesTransformerEncoder(model_dim={_int_value(params.get('model_dim', params.get('d_model')), 32)}, "
        f"num_heads={_int_value(params.get('num_heads', params.get('n_heads')), 4)}, "
        f"ff_dim={_int_value(params.get('ff_dim', params.get('d_ff')), 64)}, "
        f"layers={_int_value(params.get('layers', params.get('num_layers', params.get('n_layers'))), 2)}, "
        f"dropout={dropout})"
    )


def _per_channel_linear(node: NodeSpec) -> str:
    params = node.params
    return (
        f"PerChannelLinear(in_features={_int_value(params.get('in_features'), 32)}, "
        f"out_features={_int_value(params.get('out_features'), 8)}, "
        f"channels={_int_value(params.get('channels'), 4)})"
    )


def _linear_patch_reconstruction(node: NodeSpec) -> str:
    output_shape = node.params.get("output_shape", [])
    out_features = output_shape[-1] if isinstance(output_shape, list) and output_shape else "Plen"
    out_features = _int_value(out_features, 8)
    if "in_features" not in node.params:
        return f"nn.LazyLinear(out_features={out_features})"
    return f"nn.Linear(in_features={_int_value(node.params['in_features'], 32)}, out_features={out_features})"


@dataclass(frozen=True, slots=True)
class ComponentDefinition:
    constructor: Callable[[NodeSpec], str]
    forward: Callable[[NodeSpec, str], str] = lambda node, args: f"self.{node.id}({args})"


def _concat(node: NodeSpec, args: str) -> str:
    dim = int(node.params.get("dim", 1))
    return f"torch.cat([{args}], dim={dim})"


def _add(node: NodeSpec, args: str) -> str:
    if node.params.get("positional_encoding") == "fixed":
        return f"self.{node.id}({args})"
    return " + ".join(item.strip() for item in args.split(","))


def _add_constructor(node: NodeSpec) -> str:
    if node.params.get("positional_encoding") == "fixed":
        shape = node.params.get("shape", [])
        model_dim = shape[-1] if isinstance(shape, list) and shape else "Dmodel"
        return f"FixedPositionalEncoding(model_dim={_int_value(model_dim, 32)})"
    return "nn.Identity()"


def _reduce_sum_constructor(node: NodeSpec) -> str:
    metric = str(node.params.get("metric", "")).lower()
    return "WindowReconstructionError()" if "reconstruction" in metric else "nn.Identity()"


def _reduce_sum(node: NodeSpec, args: str) -> str:
    metric = str(node.params.get("metric", "")).lower()
    if "reconstruction" in metric:
        return f"self.{node.id}({args})"
    dimensions = node.params.get("dim")
    return f"torch.sum({args}, dim={dimensions!r})" if dimensions is not None else f"torch.sum({args})"


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
    "linear": ComponentDefinition(_linear),
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
    "add": ComponentDefinition(_add_constructor, _add),
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
    "repeatpadlastobservation": ComponentDefinition(_repeat_pad),
    "repeatpadlaststep": ComponentDefinition(_repeat_pad),
    "repeatlaststep": ComponentDefinition(_repeat_pad),
    "slidingwindowpatchify": ComponentDefinition(_patchify),
    "padandextractpatches": ComponentDefinition(_pad_and_extract_patches),
    "patchifyovertime": ComponentDefinition(_pad_and_extract_patches),
    "patching": ComponentDefinition(_pad_and_extract_patches),
    "addfixedpositionalencoding": ComponentDefinition(_fixed_position),
    "addpositionalencoding": ComponentDefinition(_fixed_position),
    "linearaddfixedpositionalencoding": ComponentDefinition(_linear_add_fixed_position),
    "linearaddpositionalencoding": ComponentDefinition(_linear_add_fixed_position),
    "transformerencoder": ComponentDefinition(_time_series_transformer),
    "transformerencoderstack": ComponentDefinition(_time_series_transformer),
    "perchannellinear": ComponentDefinition(_per_channel_linear),
    "permodalitylinear": ComponentDefinition(_per_channel_linear),
    "linearpatchreconstruction": ComponentDefinition(_linear_patch_reconstruction),
    "patchreconstructionprojection": ComponentDefinition(_linear_patch_reconstruction),
    "slicelastpatchpair": ComponentDefinition(lambda node: "LastPatchPair()"),
    "squarederrorreduce": ComponentDefinition(lambda node: "SquaredErrorReduce()"),
    "l2patchwiseerror": ComponentDefinition(lambda node: "PatchwiseL2Error()"),
    "reducesum": ComponentDefinition(_reduce_sum_constructor, _reduce_sum),
}


def normalize_op(op: str) -> str:
    return "".join(character for character in op.lower() if character.isalnum())


def definition_for(node: NodeSpec) -> ComponentDefinition:
    try:
        return REGISTRY[normalize_op(node.op)]
    except KeyError as exc:
        raise UnsupportedOperationError(f"unsupported operation: {node.op}") from exc
