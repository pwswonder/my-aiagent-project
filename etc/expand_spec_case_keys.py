# expand_spec_case_keys.py
# -*- coding: utf-8 -*-
"""
spec_scenarios.json의 blocks 키를 대/소문자 모두로 중복 등록해
코드젠이 대문자/소문자 어느 쪽을 보든 매칭되게 만든다.
실제 코드(값)는 동일 객체를 복사해 넣음. 충돌 시 원래 키를 우선.
"""
from __future__ import annotations
import json
from pathlib import Path

IN = Path("spec_scenarios.json")
OUT = Path("spec_scenarios_casefix.json")


def main():
    spec = json.loads(IN.read_text(encoding="utf-8"))
    for tpl, entry in spec.items():
        blocks = entry.get("blocks", {})
        # 원본 유지
        new_blocks = dict(blocks)
        for k, v in list(blocks.items()):
            ku, kl = k.upper(), k.lower()
            # 원래 키가 최우선, 없을 때만 보강
            if ku not in new_blocks:
                new_blocks[ku] = v
            if kl not in new_blocks:
                new_blocks[kl] = v
        entry["blocks"] = new_blocks
    OUT.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] wrote:", OUT.resolve())


if __name__ == "__main__":
    main()
