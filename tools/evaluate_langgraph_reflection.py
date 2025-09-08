
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, sys, re, difflib
from pathlib import Path
from typing import Dict, Any, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SLOT_RE = re.compile(r"^([ \t]*)#\s*(?:\{\%\s*raw\s*\%\}\s*)?\{\{CUSTOM_BLOCK:([a-zA-Z0-9_]+)\}\}(?:\s*\{\%\s*endraw\s*\%\})?\s*$", re.M,
)
def _normalize_for_diff(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.splitlines())
    return s

def _preflight(src: str) -> Tuple[bool, str]:
    try:
        compile(src, "<gen.py>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}: {e.text}"
    except Exception as e:
        return True, f"Non-fatal: {type(e).__name__}: {e}"

def _find_slot_lines(src: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for i, ln in enumerate(src.splitlines(), start=1):
        m = SLOT_RE.match(ln)
        if m:
            mapping[i] = m.group(2)
    return mapping

def _diff_changed_lines(a: str, b: str) -> List[int]:
    sm = difflib.SequenceMatcher(None, a.splitlines(True), b.splitlines(True))
    changed = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            changed.extend(list(range(i1+1, i2+1)))  # 1-based
    return changed

def _is_ignorable_baseline_line(src: str, ln_no: int) -> bool:
    lines = src.splitlines()
    if 1 <= ln_no <= len(lines):
        return lines[ln_no-1].strip() == ""
    return False

def _analyze(src: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from services.code_quality_analyzer import analyze_quality
        return analyze_quality(src, spec)
    except Exception as e:
        return {"score": 0, "issues": [{"code":"ANALYZER_ERROR","severity":"low","msg":str(e)}], "hints":[]}

def evaluate(baseline_src: str, reflected_src: str, spec: Dict[str, Any], slot_payloads: Dict[str,str], min_gain: int = 5) -> Dict[str, Any]:
    ok0, log0 = _preflight(baseline_src)
    ok1, log1 = _preflight(reflected_src)

    rep0 = _analyze(baseline_src, spec)
    rep1 = _analyze(reflected_src, spec)

    # Strict check
    slots = _find_slot_lines(baseline_src)
    changed = _diff_changed_lines(baseline_src, reflected_src)
    offenders = [ln for ln in changed if (ln not in slots) and (not _is_ignorable_baseline_line(baseline_src, ln))]
    changed_only_slots = (len(offenders) == 0)

    # Lenient check
    base_n = _normalize_for_diff(baseline_src)
    refl_n = _normalize_for_diff(reflected_src)
    slots_n = _find_slot_lines(base_n)
    changed_n = _diff_changed_lines(base_n, refl_n)
    offenders_n = [ln for ln in changed_n if (ln not in slots_n) and (not _is_ignorable_baseline_line(base_n, ln))]
    changed_only_slots_lenient = (len(offenders_n) == 0)

    sanitized_flags = {
        "no_codefence": ("```" not in reflected_src),
        "no_notebook_magic": ("%%time" not in reflected_src),
        "json_bools_converted": not any(tok in reflected_src for tok in (" true ", " false ", " null ")),
    }

    syntax_improved = (not ok0) and ok1
    score_gain = int(rep1.get("score", 0)) - int(rep0.get("score", 0))
    issues_reduced = len(rep1.get("issues", [])) < len(rep0.get("issues", []))

    payloads_non_empty = bool(slot_payloads)
    success = False
    reasons: List[str] = []
    slot_guard_ok = changed_only_slots or changed_only_slots_lenient

    if syntax_improved:
        success = True; reasons.append("syntax fixed")
    if ok1 and (score_gain >= min_gain or issues_reduced):
        success = True; reasons.append(f"quality improved (gain={score_gain}, reduced={issues_reduced})")
    if not payloads_non_empty:
        reasons.append("no slot payloads (no-op?)")
    if not slot_guard_ok:
        reasons.append("changed lines beyond slots (violation)")

    return {
        "success": success and slot_guard_ok,
        "reasons": reasons,
        "baseline": {"syntax_ok": ok0, "preflight_log": log0, "score": rep0.get("score",0), "issues": rep0.get("issues",[])},
        "reflected": {"syntax_ok": ok1, "preflight_log": log1, "score": rep1.get("score",0), "issues": rep1.get("issues",[])},
        "delta": {"score_gain": score_gain, "issues_before": len(rep0.get("issues",[])), "issues_after": len(rep1.get("issues",[]))},
        "changed_only_slots": changed_only_slots,
        "changed_only_slots_lenient": changed_only_slots_lenient,
        "sanitized": sanitized_flags,
    }

def main():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--reflected", required=True)
    ap.add_argument("--spec", required=False)
    ap.add_argument("--slot-payloads", required=False)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--min-gain", type=int, default=5)
    args = ap.parse_args()

    spec = {}
    if args.spec:
        try: spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        except Exception: pass

    slot_payloads = {}
    if args.slot_payloads:
        try:
            obj = json.loads(Path(args.slot_payloads).read_text(encoding="utf-8"))
            slot_payloads = obj.get("slot_payloads") or obj
        except Exception: pass

    base = Path(args.baseline).read_text(encoding="utf-8")
    refl = Path(args.reflected).read_text(encoding="utf-8")

    report = evaluate(base, refl, spec, slot_payloads, min_gain=args.min_gain)
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"SUCCESS: {report['success']}")
    lines.append(f"REASONS: {', '.join(report['reasons']) or '-'}")
    lines.append(f"BASE: syntax_ok={report['baseline']['syntax_ok']}, score={report['baseline']['score']}, issues={len(report['baseline']['issues'])}")
    lines.append(f"REFL: syntax_ok={report['reflected']['syntax_ok']}, score={report['reflected']['score']}, issues={len(report['reflected']['issues'])}")
    lines.append(f"DELTA: score_gain={report['delta']['score_gain']}, issues_before={report['delta']['issues_before']} -> after={report['delta']['issues_after']}")
    lines.append(f"CHANGED_ONLY_SLOTS: {report['changed_only_slots']}")
    lines.append(f"CHANGED_ONLY_SLOTS_LENIENT: {report['changed_only_slots_lenient']}")
    lines.append(f"SANITIZED: {report['sanitized']}")
    Path(args.out_txt).write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()
