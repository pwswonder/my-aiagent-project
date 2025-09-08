# make_slot_tools_bundle.py
# -*- coding: utf-8 -*-
"""
slot_tools_bundle.zip 생성기
- slot_registry.py
- template_slot_tools.py
- build_spec_scenarios.py
- README.md
를 포함한 ZIP을 현재 디렉토리에 생성합니다.
"""

from __future__ import annotations
import zipfile
from pathlib import Path
import textwrap


def main() -> None:
    # 1) 번들 디렉토리 준비
    out_dir = Path(".").resolve()
    bundle_dir = out_dir / "slot_tools_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 2) 파일 본문 구성 (필요시 수정/확장 가능)
    slot_registry_py = r"""# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Any
import json, re
from pathlib import Path

FAMILY_RULES = [
    ("Transformer", [r"MultiHeadAttention\(", r"\bTransformer\b"]),
    ("RNN",         [r"\b(LSTM|GRU|SimpleRNN)\("]),
    ("CNN",         [r"\bConv2D\("]),
]

def infer_family_from_source(src: str) -> str:
    for fam, pats in FAMILY_RULES:
        if any(re.search(p, src) for p in pats):
            return fam
    return "Generic"

def infer_task_from_family(family: str, filename: str) -> str:
    lc = filename.lower()
    if family == "Transformer":
        return "sequence_to_sequence_mt" if ("mt" in lc or "translation" in lc) else "sequence_modeling"
    if family == "RNN":
        return "seq2seq_toy" if ("seq" in lc or "toy" in lc or "mt" in lc) else "sequence_modeling"
    if family == "CNN":
        return "image_segmentation" if ("seg" in lc or "unet" in lc) else "image_classification"
    return "generic_task"

def _builtin_registry() -> Dict[str, Dict[str, List[dict]]]:
    return {
        "Transformer": {"slots": [
            {"name":"MODEL_IMPORTS",
             "contract":{"inputs":[], "provides":["keras","layers","tf"], "forbidden":["model.compile","model.fit"]},
             "example_code":"import tensorflow as tf\nfrom tensorflow import keras\nfrom tensorflow.keras import layers\n"},
            {"name":"EMBEDDING_BLOCK",
             "contract":{"inputs":["src_tokens","tgt_tokens","vocab_size_src","vocab_size_tgt","d_model"],
                         "provides":["src_emb","tgt_emb","enc_inp","dec_inp"], "forbidden":["model.compile","model.fit"]},
             "example_code":"src_emb = layers.Embedding(input_dim={vocab_size_src}, output_dim={d_model}, mask_zero=True, name='src_tok_emb')(src_tokens)\n"
                            "tgt_emb = layers.Embedding(input_dim={vocab_size_tgt}, output_dim={d_model}, mask_zero=True, name='tgt_tok_emb')(tgt_tokens)\n"
                            "enc_inp = src_emb\ndec_inp = tgt_emb\n"},
            {"name":"ENCODER_STACK_BLOCK",
             "contract":{"inputs":["enc_inp","num_encoder_layers","num_heads","d_model","ff_dim","dropout"],
                         "provides":["encoder_out"], "forbidden":["model.compile","model.fit"]},
             "example_code":"x = enc_inp\nfor i in range({num_encoder_layers}):\n"
                            "    attn = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={d_model}//{num_heads}, dropout={dropout}, name=f'enc_mha_{i}')(x, x)\n"
                            "    x = layers.Add()([x, attn])\n    x = layers.LayerNormalization(epsilon=1e-6, name=f'enc_ln1_{i}')(x)\n"
                            "    ff = layers.Dense({ff_dim}, activation='relu', name=f'enc_ff1_{i}')(x)\n"
                            "    ff = layers.Dense({d_model}, name=f'enc_ff2_{i}')(ff)\n"
                            "    x = layers.Add()([x, ff])\n    x = layers.LayerNormalization(epsilon=1e-6, name=f'enc_ln2_{i}')(x)\n"
                            "encoder_out = x\n"},
            {"name":"DECODER_STACK_BLOCK",
             "contract":{"inputs":["dec_inp","encoder_out","num_decoder_layers","num_heads","d_model","ff_dim","dropout"],
                         "provides":["decoder_out"], "forbidden":["model.compile","model.fit"]},
             "example_code":"y = dec_inp\nfor i in range({num_decoder_layers}):\n"
                            "    self_attn = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={d_model}//{num_heads}, dropout={dropout}, name=f'dec_self_mha_{i}')(y, y)\n"
                            "    y = layers.Add()([y, self_attn])\n    y = layers.LayerNormalization(epsilon=1e-6, name=f'dec_ln1_{i}')(y)\n"
                            "    cross = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={d_model}//{num_heads}, dropout={dropout}, name=f'dec_cross_mha_{i}')(y, encoder_out)\n"
                            "    y = layers.Add()([y, cross])\n    y = layers.LayerNormalization(epsilon=1e-6, name=f'dec_ln2_{i}')(y)\n"
                            "    ff = layers.Dense({ff_dim}, activation='relu', name=f'dec_ff1_{i}')(y)\n"
                            "    ff = layers.Dense({d_model}, name=f'dec_ff2_{i}')(ff)\n"
                            "    y = layers.Add()([y, ff])\n    y = layers.LayerNormalization(epsilon=1e-6, name=f'dec_ln3_{i}')(y)\n"
                            "decoder_out = y\n"},
            {"name":"HEAD_BLOCK",
             "contract":{"inputs":["decoder_out","vocab_size_tgt"], "provides":["logits"], "forbidden":[]},
             "example_code":"logits = layers.Dense({vocab_size_tgt}, name='lm_head')(decoder_out)\n"},
            {"name":"FIT_KWARGS",
             "contract":{"inputs":[], "provides":[], "forbidden":[]},
             "example_code":"callbacks=[keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max', patience=2, restore_best_weights=True)],\nvalidation_split=0.05,\nshuffle=True"}
        ]},
        "RNN": {"slots": [
            {"name":"EMBEDDING_BLOCK",
             "contract":{"inputs":["src_tokens","tgt_tokens","vocab_size_src","vocab_size_tgt","embed_dim"], "provides":["src_emb","tgt_emb"], "forbidden":[]},
             "example_code":"src_emb = layers.Embedding({vocab_size_src}, {embed_dim}, mask_zero=True, name='src_emb')(src_tokens)\n"
                            "tgt_emb = layers.Embedding({vocab_size_tgt}, {embed_dim}, mask_zero=True, name='tgt_emb')(tgt_tokens)\n"},
            {"name":"ENCODER_BLOCK",
             "contract":{"inputs":["src_emb","enc_units"], "provides":["enc_out","enc_h","enc_c"], "forbidden":[]},
             "example_code":"enc_out, enc_h, enc_c = layers.LSTM({enc_units}, return_sequences=True, return_state=True, name='encoder_lstm')(src_emb)\n"},
            {"name":"ATTN_BLOCK",
             "contract":{"inputs":["dec_lstm_out","enc_out"], "provides":["context","attn_weights"], "forbidden":[]},
             "example_code":"def luong_dot_attention(query, values):\n    scores = tf.matmul(query, values, transpose_b=True)\n    weights = tf.nn.softmax(scores, axis=-1)\n    context = tf.matmul(weights, values)\n    return context, weights\ncontext, attn_weights = luong_dot_attention(dec_lstm_out, enc_out)\n"},
            {"name":"DECODER_BLOCK",
             "contract":{"inputs":["tgt_emb","enc_h","enc_c","dec_units"], "provides":["dec_lstm_out"], "forbidden":[]},
             "example_code":"dec_lstm_out, _, _ = layers.LSTM({dec_units}, return_sequences=True, return_state=True, name='decoder_lstm')(tgt_emb, initial_state=[enc_h, enc_c])\n"},
            {"name":"HEAD_BLOCK",
             "contract":{"inputs":["dec_lstm_out","context","vocab_size_tgt"], "provides":["logits"], "forbidden":[]},
             "example_code":"dec_cat = layers.Concatenate()([dec_lstm_out, context])\nlogits = layers.Dense({vocab_size_tgt}, name='lm_head')(dec_cat)\n"},
            {"name":"FIT_KWARGS",
             "contract":{"inputs":[], "provides":[], "forbidden":[]},
             "example_code":"callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', factor=0.5, patience=1, min_lr=1e-5)],\nvalidation_split=0.1,\nshuffle=True"}
        ]},
        "CNN": {"slots": [
            {"name":"STEM_BLOCK",
             "contract":{"inputs":["inputs","initial_filters"], "provides":["x"], "forbidden":[]},
             "example_code":"x = layers.Conv2D({initial_filters}, 7, strides=2, padding='same', use_bias=False, name='stem_conv')(inputs)\n"
                            "x = layers.BatchNormalization(name='stem_bn')(x)\n"
                            "x = layers.ReLU(name='stem_relu')(x)\n"
                            "x = layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='stem_pool')(x)\n"},
            {"name":"RESIDUAL_STAGE_1",
             "contract":{"inputs":["x"], "provides":["x"], "forbidden":[]},
             "example_code":"for i in range(2):\n"
                            "    sc = x\n"
                            "    x = layers.Conv2D({initial_filters}, 3, padding='same', use_bias=False, name=f'res1_conv1_{i}')(x)\n"
                            "    x = layers.BatchNormalization(name=f'res1_bn1_{i}')(x)\n"
                            "    x = layers.ReLU(name=f'res1_relu1_{i}')(x)\n"
                            "    x = layers.Conv2D({initial_filters}, 3, padding='same', use_bias=False, name=f'res1_conv2_{i}')(x)\n"
                            "    x = layers.BatchNormalization(name=f'res1_bn2_{i}')(x)\n"
                            "    x = layers.Add(name=f'res1_add_{i}')([x, sc])\n"
                            "    x = layers.ReLU(name=f'res1_out_{i}')(x)\n"},
            {"name":"RESIDUAL_STAGE_2",
             "contract":{"inputs":["x"], "provides":["x"], "forbidden":[]},
             "example_code":"for i in range(2):\n"
                            "    sc = x\n"
                            "    x = layers.Conv2D({initial_filters}*2, 3, strides=2 if i==0 else 1, padding='same', use_bias=False, name=f'res2_conv1_{i}')(x)\n"
                            "    x = layers.BatchNormalization(name=f'res2_bn1_{i}')(x)\n"
                            "    x = layers.ReLU(name=f'res2_relu1_{i}')(x)\n"
                            "    x = layers.Conv2D({initial_filters}*2, 3, padding='same', use_bias=False, name=f'res2_conv2_{i}')(x)\n"
                            "    x = layers.BatchNormalization(name=f'res2_bn2_{i}')(x)\n"
                            "    if i==0:\n"
                            "        sc = layers.Conv2D({initial_filters}*2, 1, strides=2, padding='same', use_bias=False, name='res2_sc')(sc)\n"
                            "        sc = layers.BatchNormalization(name='res2_sc_bn')(sc)\n"
                            "    x = layers.Add(name=f'res2_add_{i}')([x, sc])\n"
                            "    x = layers.ReLU(name=f'res2_out_{i}')(x)\n"},
            {"name":"RESIDUAL_STAGE_3",
             "contract":{"inputs":["x"], "provides":["x"], "forbidden":[]},
             "example_code":"for i in range(2):\n"
                            "    sc = x\n"
                            "    x = layers.Conv2D({initial_filters}*4, 3, strides=2 if i==0 else 1, padding='same', use_bias=False, name=f'res3_conv1_{i}')(x)\n"
                            "    x = layers.BatchNormalization(name=f'res3_bn1_{i}')(x)\n"
                            "    x = layers.ReLU(name=f'res3_relu1_{i}')(x)\n"
                            "    x = layers.Conv2D({initial_filters}*4, 3, padding='same', use_bias=False, name=f'res3_conv2_{i}')(x)\n"
                            "    x = layers.BatchNormalization(name=f'res3_bn2_{i}')(x)\n"
                            "    if i==0:\n"
                            "        sc = layers.Conv2D({initial_filters}*4, 1, strides=2, padding='same', use_bias=False, name='res3_sc')(sc)\n"
                            "        sc = layers.BatchNormalization(name='res3_sc_bn')(sc)\n"
                            "    x = layers.Add(name=f'res3_add_{i}')([x, sc])\n"
                            "    x = layers.ReLU(name=f'res3_out_{i}')(x)\n"},
            {"name":"CLASSIFIER_HEAD",
             "contract":{"inputs":["x","num_classes"], "provides":["logits"], "forbidden":[]},
             "example_code":"x = layers.GlobalAveragePooling2D(name='gap')(x)\nlogits = layers.Dense({num_classes}, name='logits')(x)\n"},
            {"name":"FIT_KWARGS",
             "contract":{"inputs":[], "provides":[], "forbidden":[]},
             "example_code":"callbacks=[keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max', patience=3, restore_best_weights=True)],\nvalidation_split=0.1,\nshuffle=True"}
        ]},
        "Generic": {"slots": [
            {"name":"FIT_KWARGS",
             "contract":{"inputs":[], "provides":[], "forbidden":[]},
             "example_code":"shuffle=True\n"}
        ]}
    }

def _load_overrides() -> Dict[str, Any]:
    path = Path("slot_registry_overrides.json")
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def merge_registry(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for fam, conf in overrides.items():
        if fam not in base:
            base[fam] = conf
        else:
            if "slots" in conf:
                by_name = {s["name"]: s for s in base[fam].get("slots", [])}
                for s in conf["slots"]:
                    by_name[s["name"]] = s
                base[fam]["slots"] = list(by_name.values())
    return base

def get_slot_registry() -> Dict[str, Dict[str, Any]]:
    reg = _builtin_registry()
    overrides = _load_overrides()
    return merge_registry(reg, overrides)

__all__ = ["get_slot_registry", "infer_family_from_source", "infer_task_from_family"]
"""

    template_slot_tools_py = r'''# -*- coding: utf-8 -*-
"""
template_slot_tools.py
- Extract block names from Jinja templates
- Suggest anchors for insertion (family-aware)
- Build spec_scenarios.json with 1:1 mapping
"""
from __future__ import annotations
import re, json
from pathlib import Path
from typing import Dict, List, Any
from slot_registry import get_slot_registry, infer_family_from_source, infer_task_from_family

BLOCK_PATTERNS = [
    r"CUSTOM_BLOCK\(\s*['\"]([^'\"]+)['\"]\s*\)",
    r"AUTOBLOCK\(\s*['\"]([^'\"]+)['\"]\s*\)",
    r"AUTOBLOCK[:\s]+([A-Z0-9_]+)",
    r"CUSTOM_BLOCK[:\s]+([A-Z0-9_]+)",
]

def extract_blocks(src: str) -> List[str]:
    found = []
    for pat in BLOCK_PATTERNS:
        found += re.findall(pat, src)
    seen, out = set(), []
    for s in found:
        s = s.strip()
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out

def suggest_anchors(src: str, family: str) -> Dict[str, Dict[str, int | str]]:
    lines = src.splitlines()
    def after_last_import():
        idx = -1
        for i, ln in enumerate(lines):
            if re.match(r"\s*(import|from)\s+", ln): idx = i
        return idx + 1
    def first(pat):
        r = re.compile(pat)
        for i, ln in enumerate(lines):
            if r.search(ln): return i
        return -1
    def last(pat):
        r = re.compile(pat); idx = -1
        for i, ln in enumerate(lines):
            if r.search(ln): idx = i
        return idx

    if family == "Transformer":
        return {
            "MODEL_IMPORTS": {"line": after_last_import(), "hint":"after last import"},
            "EMBEDDING_BLOCK": {"line": first(r"Embedding\("), "hint":"before first Embedding("},
            "ENCODER_STACK_BLOCK": {"line": first(r"MultiHeadAttention\("), "hint":"before first MultiHeadAttention("},
            "DECODER_STACK_BLOCK": {"line": last(r"MultiHeadAttention\(")+1, "hint":"after last MultiHeadAttention("},
            "HEAD_BLOCK": {"line": last(r"Dense\("), "hint":"before last Dense("},
            "FIT_KWARGS": {"line": first(r"model\.fit\("), "hint":"inside model.fit(...)"}
        }
    if family == "RNN":
        return {
            "EMBEDDING_BLOCK": {"line": first(r"Embedding\("), "hint":"before first Embedding("},
            "ENCODER_BLOCK": {"line": first(r"LSTM\("), "hint":"before first LSTM("},
            "DECODER_BLOCK": {"line": last(r"LSTM\(")+1, "hint":"after last LSTM("},
            "ATTN_BLOCK": {"line": first(r"attention|Attention"), "hint":"near attention helper/section"},
            "HEAD_BLOCK": {"line": last(r"Dense\("), "hint":"before last Dense("},
            "FIT_KWARGS": {"line": first(r"model\.fit\("), "hint":"inside model.fit(...)"}
        }
    if family == "CNN":
        return {
            "STEM_BLOCK": {"line": first(r"Conv2D\("), "hint":"before first Conv2D("},
            "RESIDUAL_STAGE_1": {"line": first(r"res1|stage\s*1|Residual.*1"), "hint":"near stage-1"},
            "RESIDUAL_STAGE_2": {"line": first(r"res2|stage\s*2|Residual.*2"), "hint":"near stage-2"},
            "RESIDUAL_STAGE_3": {"line": first(r"res3|stage\s*3|Residual.*3"), "hint":"near stage-3"},
            "CLASSIFIER_HEAD": {"line": last(r"Dense\("), "hint":"before last Dense("},
            "FIT_KWARGS": {"line": first(r"model\.fit\("), "hint":"inside model.fit(...)"}
        }
    return {"FIT_KWARGS": {"line": first(r"model\.fit\("), "hint":"inside model.fit(...)"}}  # Generic

def build_scenarios(templates_dir: str, out_json: str) -> Dict[str, Any]:
    reg = get_slot_registry()
    tdir = Path(templates_dir)
    result: Dict[str, Any] = {}
    for path in sorted(tdir.glob("**/*.j2")):
        key = path.stem
        src = path.read_text(encoding="utf-8")
        family = infer_family_from_source(src)
        task = infer_task_from_family(family, path.name)
        present = extract_blocks(src)
        # 권장 레지스트리에서 현재 슬롯만 매핑
        by_name = {s["name"]: s for s in reg.get(family, {}).get("slots", [])}
        blocks = {}
        for name in present:
            if name in by_name:
                blocks[name] = {"params": {}, "code": by_name[name].get("example_code", "")}
            else:
                blocks[name] = {"params": {}, "code": f"# TODO: provide code for slot {name}\n"}
        meta = {
            "template_key": key, "family": family, "framework": "tensorflow.keras", "task": task,
            "model": {}, "training": {"epochs": 3, "batch_size": 32}, "compile": {}, "fit": {},
            "blocks": blocks
        }
        result[key] = meta
    Path(out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

def write_inventory(templates_dir: str, out_json: str) -> Dict[str, Any]:
    inv = {}
    tdir = Path(templates_dir)
    for path in sorted(tdir.glob("**/*.j2")):
        key = path.stem
        src = path.read_text(encoding="utf-8")
        inv[key] = {
            "path": str(path),
            "family": infer_family_from_source(src),
            "slots": extract_blocks(src)
        }
    Path(out_json).write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")
    return inv
'''

    build_spec_scenarios_py = r"""# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from template_slot_tools import build_scenarios, write_inventory

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates-dir", required=True)
    ap.add_argument("--out", default="spec_scenarios.json")
    ap.add_argument("--inv-out", default="block_inventory.json")
    args = ap.parse_args()
    write_inventory(args.templates_dir, args.inv_out)
    res = build_scenarios(args.templates_dir, args.out)
    print("[OK] Wrote:", args.inv_out, "and", args.out)
    print(json.dumps({k: list(v["blocks"].keys()) for k, v in res.items()}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
"""

    readme_md = textwrap.dedent(
        """\
    # Slot Tools Bundle

    ## Files
    - `slot_registry.py`: Family-aware slot taxonomy + contracts + example code. Supports local overrides via `slot_registry_overrides.json`.
    - `template_slot_tools.py`: Scans Jinja templates for slots, suggests anchors, and builds 1:1 `spec_scenarios.json`.
    - `build_spec_scenarios.py`: CLI wrapper to write `block_inventory.json` and `spec_scenarios.json`.

    ## Usage
    ```bash
    python build_spec_scenarios.py \
      --templates-dir templates_scaffolded/services \
      --out spec_scenarios.json \
      --inv-out block_inventory.json
    ```
    """
    )

    # 3) 파일 쓰기
    (bundle_dir / "slot_registry.py").write_text(slot_registry_py, encoding="utf-8")
    (bundle_dir / "template_slot_tools.py").write_text(
        template_slot_tools_py, encoding="utf-8"
    )
    (bundle_dir / "build_spec_scenarios.py").write_text(
        build_spec_scenarios_py, encoding="utf-8"
    )
    (bundle_dir / "README.md").write_text(readme_md, encoding="utf-8")

    # 4) ZIP 생성
    zip_path = Path("slot_tools_bundle.zip").resolve()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in [
            "slot_registry.py",
            "template_slot_tools.py",
            "build_spec_scenarios.py",
            "README.md",
        ]:
            zf.write(bundle_dir / p, arcname=p)

    print(f"[OK] Created: {zip_path}")


if __name__ == "__main__":
    main()
