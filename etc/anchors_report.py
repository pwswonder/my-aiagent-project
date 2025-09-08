# anchors_report.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from template_slot_tools_v2 import suggest_anchors, extract_blocks
from slot_registry import infer_family_from_source

def main(templates_dir: str, out_json: str = "anchors_report.json"):
    res = {}
    for p in sorted(Path(templates_dir).glob("**/*.j2")):
        src = p.read_text(encoding="utf-8")
        fam = infer_family_from_source(src)
        anchors = suggest_anchors(src, fam)
        blocks = extract_blocks(src)
        res[p.stem] = {"family": fam, "anchors": anchors, "found_blocks": blocks}
    Path(out_json).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] wrote", out_json)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates-dir", required=True)
    ap.add_argument("--out", default="anchors_report.json")
    args = ap.parse_args()
    main(args.templates_dir, args.out)
