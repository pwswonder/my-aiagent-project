# inject_slots_copy.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import List
from slot_registry import get_slot_registry, infer_family_from_source
from template_slot_tools_v2 import extract_blocks, suggest_anchors


def make_raw_line(slot: str) -> str:
    # Jinja 리터럴을 안전하게 출력하기 위해 중괄호는 이중으로, raw/endraw는 그대로 감싼다.
    return f"# {{% raw %}}{{{{CUSTOM_BLOCK:{slot}}}}}{{% endraw %}}\n"


def insert_at(lines: List[str], idx: int, snippet: str) -> List[str]:
    if idx < 0:
        idx = 0
    if idx > len(lines):
        idx = len(lines)
    return lines[:idx] + [snippet] + lines[idx:]


def main(templates_dir: str, out_dir: str, only_missing: bool, apply: bool):
    tdir, odir = Path(templates_dir), Path(out_dir)
    odir.mkdir(parents=True, exist_ok=True)
    reg = get_slot_registry()
    for src_path in sorted(tdir.glob("**/*.j2")):
        text = src_path.read_text(encoding="utf-8")
        fam = infer_family_from_source(text)
        anchors = suggest_anchors(text, fam)
        existing = set(extract_blocks(text))
        # 후보 슬롯: 레지스트리 등록 슬롯
        cand = [s["name"] for s in reg.get(fam, {}).get("slots", [])]
        # only_missing이면 기존에 없는 슬롯만
        targets = [c for c in cand if (c not in existing)] if only_missing else cand[:]
        lines = text.splitlines(True)
        # 간단 삽입 정책: 각 target에 대해 anchor 있으면 그 라인에 삽입, 없으면 파일 끝
        for slot in targets:
            anchor = anchors.get(slot, {})
            line_idx = anchor.get("line", len(lines))
            lines = insert_at(lines, line_idx, make_raw_line(slot))
        out_path = odir / src_path.name
        if apply:
            out_path.write_text("".join(lines), encoding="utf-8")
        else:
            # 드라이런: 파일 헤더에 미리보기 헤더만 남김
            preview = ["# ---- DRYRUN PREVIEW: will insert below lines ----\n"] + lines
            out_path.write_text("".join(preview), encoding="utf-8")
        print("[OK] wrote", out_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--templates-dir", required=True)
    ap.add_argument("--out-dir", default="templates_scaffolded_slots")
    ap.add_argument("--only-missing", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    main(args.templates_dir, args.out_dir, args.only_missing, args.apply)
