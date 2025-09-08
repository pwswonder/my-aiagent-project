# debug_missing_baseline.py
# -*- coding: utf-8 -*-
"""
- spec_scenarios.json에 각 템플릿 키가 있는지/블록 개수는 얼마인지 확인
- 템플릿 내 CUSTOM_BLOCK 패턴이 'raw' 기반/비-raw 기반 중 무엇에 매칭되는지 검사
- 어떤 이유로 baseline이 안 나왔는지 텍스트로 요약
"""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, List

# 필요시 아래 두 경로를 현재 환경에 맞게 수정
SCENARIO = Path("spec_scenarios.json")  # C 단계 산출물
TPL_ROOT = Path("templates_scaffolded_slots")  # D를 했다면 slots/ , 안했다면 services/

RAW_RE = re.compile(
    r"(?m)^\s*#\s*\{\%\s*raw\s*\%\}\{\{CUSTOM_BLOCK:([A-Za-z0-9_]+)\}\}\{\%\s*endraw\s*\%\}\s*$"
)
NORAW_RE = re.compile(r"(?m)^\s*#\s*\{\{CUSTOM_BLOCK:([A-Za-z0-9_\-]+)\}\}\s*$")


def find_blocks(text: str) -> Dict[str, List[str]]:
    raw = RAW_RE.findall(text)
    noraw = NORAW_RE.findall(text)
    return {"raw": raw, "noraw": noraw}


def main():
    if not SCENARIO.exists():
        print(f"[ERR] spec file not found: {SCENARIO.resolve()}")
        return
    spec = json.loads(SCENARIO.read_text(encoding="utf-8"))

    rows = []
    for p in sorted(TPL_ROOT.glob("*.j2")):
        key = p.stem
        entry = spec.get(key)
        present = bool(entry)
        n_blocks = len((entry or {}).get("blocks", {})) if present else 0

        txt = p.read_text(encoding="utf-8")
        kinds = find_blocks(txt)
        raw_cnt, noraw_cnt = len(kinds["raw"]), len(kinds["noraw"])

        reason = []
        if not present:
            reason.append("C단계 시나리오에 템플릿 키 없음")
        elif n_blocks == 0:
            reason.append("시나리오 blocks 비어 있음(치환 코드 0)")
        if raw_cnt == 0 and noraw_cnt == 0:
            reason.append("템플릿에 CUSTOM_BLOCK 주석 줄 자체가 없음")
        elif raw_cnt > 0 and noraw_cnt == 0:
            reason.append("템플릿은 raw 패턴(# {% raw %}..{% endraw %})만 있음")
        elif noraw_cnt > 0 and raw_cnt == 0:
            reason.append("템플릿은 비-raw 패턴(# {{CUSTOM_BLOCK:..}})만 있음")
        else:
            reason.append("두 패턴이 혼재")

        rows.append(
            {
                "template": key,
                "in_spec": present,
                "spec_block_count": n_blocks,
                "tmpl_raw_blocks": raw_cnt,
                "tmpl_noraw_blocks": noraw_cnt,
                "note": "; ".join(reason),
            }
        )

    # 깔끔 출력
    print("template | in_spec | spec_blocks | tmpl_raw | tmpl_noraw | note")
    for r in rows:
        print(
            f"{r['template']:15} | {str(r['in_spec']):7} | {r['spec_block_count']:11} |"
            f" {r['tmpl_raw_blocks']:8} | {r['tmpl_noraw_blocks']:10} | {r['note']}"
        )


if __name__ == "__main__":
    main()
