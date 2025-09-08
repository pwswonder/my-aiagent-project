
# -*- coding: utf-8 -*-
"""
tools/run_reflection_regression.py

Run the existing e2e pipeline (codegen -> langgraph reflection -> evaluate)
across multiple template keys, collect results, and emit a summary table.

Usage:
  python tools/run_reflection_regression.py     --templates resnet transformer transformer_mt unet rnn_seq cnn_family autoencoder vae gan     --outroot ./e2e_regression     --rounds 2

Notes:
- This script shells out to tools/run_reflections_e2e_smoke_v2.py for each key.
- Pass/fail rule (default):
    success == True AND changed_only_slots == True AND issues_after <= 1
- Outputs per-template artifacts under {outroot}/{template_key}/
- Also writes:
    {outroot}/summary.json
    {outroot}/summary.csv
    {outroot}/summary.md
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "tools" / "run_reflections_e2e_smoke_v2.py"

def run_one(template_key: str, outroot: Path, rounds: int) -> Dict:
    outdir = outroot / template_key
    outdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LG_REFLECTION_ROUNDS"] = str(rounds)
    # Call the official e2e runner
    cmd = [sys.executable, str(E2E), "--template-key", template_key, "--outdir", str(outdir)]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    # Attempt to read the produced JSON
    json_path = outdir / "02_langgraph_eval.json"
    ok = json_path.exists()
    payload = {}
    if ok:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            payload = {"success": False, "error": f"json parse error: {e}"}
    else:
        payload = {"success": False, "error": f"missing {json_path.name}", "stdout": res.stdout[-4000:], "stderr": res.stderr[-4000:]}
    return {
        "template_key": template_key,
        "outdir": str(outdir),
        "returncode": res.returncode,
        "stdout_tail": res.stdout[-2000:],
        "stderr_tail": res.stderr[-2000:],
        "report": payload,
    }

def decide_pass(report: Dict, issues_threshold: int = 1) -> bool:
    try:
        if not report.get("success"):
            return False
        if report.get("changed_only_slots") is False:
            return False
        delta = report.get("delta", {})
        after = int(delta.get("issues_after", 99))
        return after <= issues_threshold
    except Exception:
        return False

def to_csv(rows: List[Dict]) -> str:
    headers = ["template_key", "success", "score_gain", "issues_before", "issues_after", "changed_only_slots", "outdir"]
    out = [",".join(headers)]
    for r in rows:
        rep = r.get("report", {})
        delta = rep.get("delta", {})
        line = [
            r["template_key"],
            str(rep.get("success", False)),
            str(delta.get("score_gain", "")),
            str(delta.get("issues_before", "")),
            str(delta.get("issues_after", "")),
            str(rep.get("changed_only_slots", "")),
            r["outdir"],
        ]
        out.append(",".join(line))
    return "\n".join(out)

def to_md(rows: List[Dict]) -> str:
    out = ["| template | success | gain | before | after | slots-only | outdir |",
           "|---|---:|---:|---:|---:|:---:|---|"]
    for r in rows:
        rep = r.get("report", {})
        delta = rep.get("delta", {})
        out.append("| {t} | {s} | {g} | {b} | {a} | {c} | {o} |".format(
            t=r["template_key"],
            s="✅" if rep.get("success") else "❌",
            g=str(delta.get("score_gain", "")),
            b=str(delta.get("issues_before", "")),
            a=str(delta.get("issues_after", "")),
            c="✅" if rep.get("changed_only_slots") else "❌",
            o=r["outdir"],
        ))
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", nargs="+", required=False,
                    default=["resnet", "transformer", "transformer_mt", "unet", "rnn_seq", "cnn_family", "autoencoder", "vae", "gan"])
    ap.add_argument("--outroot", default="e2e_regression")
    ap.add_argument("--rounds", type=int, default=int(os.environ.get("LG_REFLECTION_ROUNDS", "2")))
    args = ap.parse_args()

    outroot = Path(args.outroot).resolve()
    outroot.mkdir(parents=True, exist_ok=True)

    rows = []
    for key in args.templates:
        print(f"[RUN] template={key}")
        result = run_one(key, outroot, args.rounds)
        rows.append(result)

    # Write summaries
    json.dump(rows, (outroot/"summary.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    (outroot/"summary.csv").write_text(to_csv(rows), encoding="utf-8")
    (outroot/"summary.md").write_text(to_md(rows), encoding="utf-8")

    # Console summary
    passes = 0
    for r in rows:
        rep = r.get("report", {})
        ok = decide_pass(rep)
        passes += int(ok)
        delta = rep.get("delta", {})
        print(f"[{r['template_key']}] success={rep.get('success')} gain={delta.get('score_gain')} after={delta.get('issues_after')} slots_only={rep.get('changed_only_slots')} → {'PASS' if ok else 'FAIL'}")
    print(f"\nPassed {passes}/{len(rows)} templates. Artifacts under: {outroot}\n")

if __name__ == "__main__":
    main()
