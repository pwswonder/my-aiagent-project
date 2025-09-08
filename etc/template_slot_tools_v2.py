# template_slot_tools_v2.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import re, json
from pathlib import Path
from typing import Dict, List, Any
from slot_registry import (
    get_slot_registry,
    infer_family_from_source,
    infer_task_from_family,
)

# {{ CUSTOM_BLOCK:foo }} / {{ CUSTOM_BLOCK("foo") }} / 텍스트 내 CUSTOM_BLOCK:foo 등 모두 허용
BLOCK_REGEXES = [
    re.compile(
        r"\{\{\s*CUSTOM_BLOCK\s*[:(]\s*['\"]?(?P<name>[A-Za-z0-9_]+)['\"]?\s*\)?\s*\}\}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\{\{\s*AUTOBLOCK\s*[:(]\s*['\"]?(?P<name>[A-Za-z0-9_]+)['\"]?\s*\)?\s*\}\}",
        re.IGNORECASE,
    ),
    re.compile(
        r"CUSTOM_BLOCK\s*[:(]\s*['\"]?(?P<name>[A-Za-z0-9_]+)['\"]?\s*\)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"AUTOBLOCK\s*[:(]\s*['\"]?(?P<name>[A-Za-z0-9_]+)['\"]?\s*\)?", re.IGNORECASE
    ),
]


def extract_blocks(src: str) -> List[str]:
    found: List[str] = []
    for rx in BLOCK_REGEXES:
        found += [m.group("name") for m in rx.finditer(src)]
    # 중복 제거(순서 보존)
    uniq, seen = [], set()
    for s in found:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def suggest_anchors(src: str, family: str) -> Dict[str, Dict[str, int | str]]:
    # 간단한 휴리스틱 앵커 (필요시 더 정교하게 확장 가능)
    lines = src.splitlines()

    def after_last_import():
        idx = -1
        for i, ln in enumerate(lines):
            if re.match(r"\s*(import|from)\s+", ln):
                idx = i
        return idx + 1

    def first(pat):
        r = re.compile(pat)
        for i, ln in enumerate(lines):
            if r.search(ln):
                return i
        return -1

    def last(pat):
        r = re.compile(pat)
        idx = -1
        for i, ln in enumerate(lines):
            if r.search(ln):
                idx = i
        return idx

    if family == "Transformer":
        return {
            "MODEL_IMPORTS": {"line": after_last_import(), "hint": "after last import"},
            "EMBEDDING_BLOCK": {
                "line": first(r"Embedding\("),
                "hint": "before first Embedding(",
            },
            "ENCODER_STACK_BLOCK": {
                "line": first(r"MultiHeadAttention\("),
                "hint": "before first MultiHeadAttention(",
            },
            "DECODER_STACK_BLOCK": {
                "line": last(r"MultiHeadAttention\(") + 1,
                "hint": "after last MultiHeadAttention(",
            },
            "HEAD_BLOCK": {"line": last(r"Dense\("), "hint": "before last Dense("},
            "FIT_KWARGS": {
                "line": first(r"model\.fit\("),
                "hint": "inside model.fit(...)",
            },
        }
    if family == "RNN":
        return {
            "EMBEDDING_BLOCK": {
                "line": first(r"Embedding\("),
                "hint": "before first Embedding(",
            },
            "ENCODER_BLOCK": {"line": first(r"LSTM\("), "hint": "before first LSTM("},
            "DECODER_BLOCK": {"line": last(r"LSTM\(") + 1, "hint": "after last LSTM("},
            "ATTN_BLOCK": {
                "line": first(r"attention|Attention"),
                "hint": "near attention helper/section",
            },
            "HEAD_BLOCK": {"line": last(r"Dense\("), "hint": "before last Dense("},
            "FIT_KWARGS": {
                "line": first(r"model\.fit\("),
                "hint": "inside model.fit(...)",
            },
        }
    if family == "CNN":
        return {
            "STEM_BLOCK": {"line": first(r"Conv2D\("), "hint": "before first Conv2D("},
            "RESIDUAL_STAGE_1": {
                "line": first(r"res1|stage\s*1|Residual.*1"),
                "hint": "near stage-1",
            },
            "RESIDUAL_STAGE_2": {
                "line": first(r"res2|stage\s*2|Residual.*2"),
                "hint": "near stage-2",
            },
            "RESIDUAL_STAGE_3": {
                "line": first(r"res3|stage\s*3|Residual.*3"),
                "hint": "near stage-3",
            },
            "CLASSIFIER_HEAD": {"line": last(r"Dense\("), "hint": "before last Dense("},
            "FIT_KWARGS": {
                "line": first(r"model\.fit\("),
                "hint": "inside model.fit(...)",
            },
        }
    return {
        "FIT_KWARGS": {"line": first(r"model\.fit\("), "hint": "inside model.fit(...)"}
    }  # Generic


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
        # 권장 레지스트리와 매핑: 템플릿에 "존재하는" 슬롯만 1:1로
        by_name = {s["name"]: s for s in reg.get(family, {}).get("slots", [])}
        blocks = {}
        for name in present:
            ex = by_name.get(name, {})
            code = ex.get("example_code", f"# TODO: provide code for slot {name}\n")
            blocks[name] = {"params": {}, "code": code}
        meta = {
            "template_key": key,
            "family": family,
            "framework": "tensorflow.keras",
            "task": task,
            "model": {},
            "training": {"epochs": 3, "batch_size": 32},
            "compile": {},
            "fit": {},
            "blocks": blocks,
        }
        result[key] = meta
    Path(out_json).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def write_inventory(templates_dir: str, out_json: str) -> Dict[str, Any]:
    inv = {}
    tdir = Path(templates_dir)
    for path in sorted(tdir.glob("**/*.j2")):
        src = path.read_text(encoding="utf-8")
        inv[path.stem] = {
            "path": str(path),
            "family": infer_family_from_source(src),
            "slots": extract_blocks(src),
        }
    Path(out_json).write_text(
        json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return inv
