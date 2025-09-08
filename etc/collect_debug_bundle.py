# collect_debug_bundle.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path

CORE_CODE = [
    "run_e2e_reflection_eval.py",
    "codegen.py",
    "codegen_autoblocks.py",
    "llm_codegen_assist.py",
    "code_quality_analyzer.py",
    "quality_reflection.py",
    "langgraph_reflection.py",
]


def safe_copy(src: Path, dst: Path, log: list[str]):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    else:
        log.append(f"[MISS] {src}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tpl-root", required=True, help="templates_scaffolded_slots or services"
    )
    ap.add_argument(
        "--templates", nargs="+", required=True, help="e.g. resnet.j2 cnn_family.j2 ..."
    )
    ap.add_argument("--spec", default="spec_scenarios_casefix.json")
    ap.add_argument("--overrides", default="slot_registry_overrides.json")
    ap.add_argument("--out", default="debug_bundle.zip")
    args = ap.parse_args()

    root = Path(".").resolve()
    outdir = root / "_debug_bundle"
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log: list[str] = []
    # 1) spec & overrides
    for p in [Path(args.spec), Path(args.overrides)]:
        safe_copy(p, outdir / p.name, log)

    # 2) core code
    for f in CORE_CODE:
        safe_copy(root / f, outdir / "code" / f, log)

    # 3) e2e outputs per template
    for t in args.templates:
        key = t.replace(".j2", "")
        base = root / "e2e_eval_out" / key
        # pick files/folders of interest
        picks = [
            "00_original.j2",
            "01_baseline_slot_applied.py",
            "02_preseed_for_reflection.py",
            "02_langgraph_eval.json",
            "03_after_langgraph.py",
            "04_after_quality_reflection.py",
        ]
        for name in picks:
            safe_copy(base / name, outdir / "e2e_eval_out" / key / name, log)
        # folders
        for name in ["10~13_diff", "."]:
            p = base / name
            if p.exists() and p.is_dir() and name != ".":
                safe_copy(p, outdir / "e2e_eval_out" / key / name, log)
        # slot regions
        for p in base.glob("20~21_slot_regions*.txt"):
            safe_copy(p, outdir / "e2e_eval_out" / key / p.name, log)

        # corresponding template (tpl-root)
        tpl = Path(args.tpl - root) / t
        safe_copy(tpl, outdir / "templates" / t, log)

    # 4) global summary & reports (optional)
    for extra in [
        "e2e_eval_out/SUMMARY.json",
        "anchors_report.json",
        "block_inventory.json",
        "slot_coverage_report.json",
    ]:
        safe_copy(root / extra, outdir / extra, log)

    # 5) write a manifest
    (outdir / "MANIFEST.txt").write_text(
        "\n".join(log) if log else "[OK] all requested files collected",
        encoding="utf-8",
    )

    # 6) zip
    shutil.make_archive(Path(args.out).with_suffix("").as_posix(), "zip", outdir)
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERR]", e)
        sys.exit(1)
