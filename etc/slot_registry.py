# slot_registry.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Any
import json, re
from pathlib import Path

FAMILY_RULES = [
    ("Transformer", [r"MultiHeadAttention\(", r"\bTransformer\b"]),
    ("RNN", [r"\b(LSTM|GRU|SimpleRNN)\("]),
    ("CNN", [r"\bConv2D\(", r"\bConv2DTranspose\("]),
]


def infer_family_from_source(src: str) -> str:
    for fam, pats in FAMILY_RULES:
        if any(re.search(p, src) for p in pats):
            return fam
    return "Generic"


def infer_task_from_family(family: str, filename: str) -> str:
    lc = filename.lower()
    if "gan" in lc:
        return "image_generation_gan"
    if "vae" in lc:
        return "image_generation_vae"
    if "autoencoder" in lc:
        return "autoencoding"
    if "unet" in lc:
        return "image_segmentation"
    if family == "Transformer":
        return (
            "sequence_to_sequence_mt"
            if ("mt" in lc or "translation" in lc)
            else "sequence_modeling"
        )
    if family == "RNN":
        return (
            "seq2seq_toy"
            if ("seq" in lc or "toy" in lc or "mt" in lc)
            else "sequence_modeling"
        )
    if family == "CNN":
        return "image_classification"
    return "generic_task"


def _builtin_registry() -> Dict[str, Dict[str, List[dict]]]:
    # 예시 코드들은 최소 예시(placeholder)이며 필요에 맞게 확장 가능
    return {
        "Transformer": {
            "slots": [
                {
                    "name": "MODEL_IMPORTS",
                    "contract": {
                        "inputs": [],
                        "provides": ["keras", "layers", "tf"],
                        "forbidden": ["model.compile", "model.fit"],
                    },
                    "example_code": "import tensorflow as tf\nfrom tensorflow import keras\nfrom tensorflow.keras import layers\n",
                },
                {
                    "name": "EMBEDDING_BLOCK",
                    "contract": {
                        "inputs": [
                            "src_tokens",
                            "tgt_tokens",
                            "vocab_size_src",
                            "vocab_size_tgt",
                            "d_model",
                        ],
                        "provides": ["src_emb", "tgt_emb", "enc_inp", "dec_inp"],
                        "forbidden": ["model.compile", "model.fit"],
                    },
                    "example_code": "src_emb = layers.Embedding(input_dim={vocab_size_src}, output_dim={d_model}, mask_zero=True)(src_tokens)\n"
                    "tgt_emb = layers.Embedding(input_dim={vocab_size_tgt}, output_dim={d_model}, mask_zero=True)(tgt_tokens)\n"
                    "enc_inp = src_emb\ndec_inp = tgt_emb\n",
                },
                {
                    "name": "ENCODER_STACK_BLOCK",
                    "contract": {
                        "inputs": [
                            "enc_inp",
                            "num_encoder_layers",
                            "num_heads",
                            "d_model",
                            "ff_dim",
                            "dropout",
                        ],
                        "provides": ["encoder_out"],
                        "forbidden": ["model.compile", "model.fit"],
                    },
                    "example_code": "# TODO: encoder stack\n",
                },
                {
                    "name": "DECODER_STACK_BLOCK",
                    "contract": {
                        "inputs": [
                            "dec_inp",
                            "encoder_out",
                            "num_decoder_layers",
                            "num_heads",
                            "d_model",
                            "ff_dim",
                            "dropout",
                        ],
                        "provides": ["decoder_out"],
                        "forbidden": ["model.compile", "model.fit"],
                    },
                    "example_code": "# TODO: decoder stack\n",
                },
                {
                    "name": "HEAD_BLOCK",
                    "contract": {
                        "inputs": ["decoder_out", "vocab_size_tgt"],
                        "provides": ["logits"],
                        "forbidden": [],
                    },
                    "example_code": "logits = layers.Dense({vocab_size_tgt})(decoder_out)\n",
                },
                {
                    "name": "FIT_KWARGS",
                    "contract": {"inputs": [], "provides": [], "forbidden": []},
                    "example_code": "shuffle=True\n",
                },
            ]
        },
        "RNN": {
            "slots": [
                {
                    "name": "EMBEDDING_BLOCK",
                    "contract": {
                        "inputs": [
                            "src_tokens",
                            "tgt_tokens",
                            "vocab_size_src",
                            "vocab_size_tgt",
                            "embed_dim",
                        ],
                        "provides": ["src_emb", "tgt_emb"],
                        "forbidden": [],
                    },
                    "example_code": "# TODO: rnn embeddings\n",
                },
                {
                    "name": "ENCODER_BLOCK",
                    "contract": {
                        "inputs": ["src_emb", "enc_units"],
                        "provides": ["enc_out", "enc_h", "enc_c"],
                        "forbidden": [],
                    },
                    "example_code": "# TODO: rnn encoder\n",
                },
                {
                    "name": "ATTN_BLOCK",
                    "contract": {
                        "inputs": ["dec_lstm_out", "enc_out"],
                        "provides": ["context", "attn_weights"],
                        "forbidden": [],
                    },
                    "example_code": "# TODO: rnn attention\n",
                },
                {
                    "name": "DECODER_BLOCK",
                    "contract": {
                        "inputs": ["tgt_emb", "enc_h", "enc_c", "dec_units"],
                        "provides": ["dec_lstm_out"],
                        "forbidden": [],
                    },
                    "example_code": "# TODO: rnn decoder\n",
                },
                {
                    "name": "HEAD_BLOCK",
                    "contract": {
                        "inputs": ["dec_lstm_out", "context", "vocab_size_tgt"],
                        "provides": ["logits"],
                        "forbidden": [],
                    },
                    "example_code": "# TODO: rnn head\n",
                },
                {
                    "name": "FIT_KWARGS",
                    "contract": {"inputs": [], "provides": [], "forbidden": []},
                    "example_code": "shuffle=True\n",
                },
            ]
        },
        "CNN": {
            "slots": [
                {
                    "name": "STEM_BLOCK",
                    "contract": {
                        "inputs": ["inputs", "initial_filters"],
                        "provides": ["x"],
                        "forbidden": [],
                    },
                    "example_code": "# TODO: cnn stem\n",
                },
                {
                    "name": "RESIDUAL_STAGE_1",
                    "contract": {"inputs": ["x"], "provides": ["x"], "forbidden": []},
                    "example_code": "# TODO: cnn stage1\n",
                },
                {
                    "name": "RESIDUAL_STAGE_2",
                    "contract": {"inputs": ["x"], "provides": ["x"], "forbidden": []},
                    "example_code": "# TODO: cnn stage2\n",
                },
                {
                    "name": "RESIDUAL_STAGE_3",
                    "contract": {"inputs": ["x"], "provides": ["x"], "forbidden": []},
                    "example_code": "# TODO: cnn stage3\n",
                },
                {
                    "name": "CLASSIFIER_HEAD",
                    "contract": {
                        "inputs": ["x", "num_classes"],
                        "provides": ["logits"],
                        "forbidden": [],
                    },
                    "example_code": "# TODO: cnn head\n",
                },
                {
                    "name": "FIT_KWARGS",
                    "contract": {"inputs": [], "provides": [], "forbidden": []},
                    "example_code": "shuffle=True\n",
                },
            ]
        },
        "Generic": {
            "slots": [
                {
                    "name": "FIT_KWARGS",
                    "contract": {"inputs": [], "provides": [], "forbidden": []},
                    "example_code": "shuffle=True\n",
                }
            ]
        },
    }


def _load_overrides() -> Dict[str, Any]:
    p = Path("slot_registry_overrides.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def merge_registry(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for fam, conf in overrides.items():
        if fam not in base:
            base[fam] = conf
        elif "slots" in conf:
            by_name = {s["name"]: s for s in base[fam].get("slots", [])}
            for s in conf["slots"]:
                by_name[s["name"]] = s
            base[fam]["slots"] = list(by_name.values())
    return base


def get_slot_registry() -> Dict[str, Dict[str, Any]]:
    reg = _builtin_registry()
    return merge_registry(reg, _load_overrides())


__all__ = ["get_slot_registry", "infer_family_from_source", "infer_task_from_family"]
