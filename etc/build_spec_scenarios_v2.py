# build_spec_scenarios_v2.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from template_slot_tools_v2 import build_scenarios, write_inventory

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates-dir", required=True)
    ap.add_argument("--out", default="spec_scenarios.json")
    ap.add_argument("--inv-out", default="block_inventory.json")
    args = ap.parse_args()
    write_inventory(args.templates_dir, args.inv_out)
    res = build_scenarios(args.templates_dir, args.out)
    print("[OK] wrote:", args.inv_out, "and", args.out)
    print(json.dumps({k: list(v["blocks"].keys()) for k, v in res.items()}, ensure_ascii=False, indent=2))
