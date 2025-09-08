# -*- coding: utf-8 -*-
"""
E2E 러너 v2: Codegen → LangGraph Reflection → 평가(자동 리포트)
- 산출물: 코드/디프/프리플라이트/평가 리포트
"""
from __future__ import annotations
import os, sys, json, difflib
from pathlib import Path

# import services.langgraph_reflection as _lg

# print("[LG] using:", _lg.__file__)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def _write(p: Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def _diff(a: str, b: str, an: str, bn: str) -> str:
    return "".join(
        difflib.unified_diff(
            a.splitlines(True), b.splitlines(True), fromfile=an, tofile=bn
        )
    )


def _preflight(src: str):
    try:
        compile(src, "<gen.py>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}: {e.text}"
    except Exception as e:
        return True, f"Non-fatal: {type(e).__name__}: {e}"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--template-key", required=True)
    ap.add_argument("--outdir", default="./e2e_artifacts")
    ap.add_argument("--spec-json", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    spec = {
        "proposed_model_family": args.template_key,
        "task_type": "Image_Classification",
        "optimizer_name": "adam",
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"],
        "dropout": 0.1,
    }
    if args.spec_json and Path(args.spec_json).exists():
        spec.update(json.loads(Path(args.spec_json).read_text(encoding="utf-8")))

    from services.basecode_service import generate_base_code

    py_path, py_src, msum = generate_base_code(args.template_key, spec)
    _write(outdir / "01_codegen_source.py", py_src)
    ok, log = _preflight(py_src)
    _write(outdir / "01_codegen_preflight.log", log or "OK")

    # LangGraph reflection
    lg_payload_path = outdir / "02_langgraph_payloads.json"
    from services.langgraph_reflection import run_langgraph_reflection

    lg = run_langgraph_reflection(
        py_src, spec, max_rounds=int(os.getenv("LG_REFLECTION_ROUNDS", "2"))
    )
    py_src_lg = lg.get("src", py_src)
    _write(outdir / "02_langgraph_reflected.py", py_src_lg)
    _write(
        outdir / "02_langgraph_diff.patch",
        _diff(py_src, py_src_lg, "codegen.py", "langgraph.py"),
    )
    _write(
        lg_payload_path,
        json.dumps(
            {"slot_payloads": lg.get("slot_payloads", {})}, ensure_ascii=False, indent=2
        ),
    )
    ok2, log2 = _preflight(py_src_lg)
    _write(outdir / "02_langgraph_preflight.log", log2 or "OK")

    # Evaluate
    from tools.evaluate_langgraph_reflection import evaluate

    report = evaluate(
        baseline_src=py_src,
        reflected_src=py_src_lg,
        spec=spec,
        slot_payloads=(
            json.loads(lg_payload_path.read_text(encoding="utf-8"))
            if lg_payload_path.exists()
            else {}
        ),
        min_gain=int(os.getenv("LG_MIN_SCORE_GAIN", "5")),
    )
    _write(
        outdir / "02_langgraph_eval.json",
        json.dumps(report, ensure_ascii=False, indent=2),
    )
    lines = [
        f"SUCCESS: {report['success']}",
        f"REASONS: {', '.join(report['reasons']) or '-'}",
        f"BASE: syntax_ok={report['baseline']['syntax_ok']}, score={report['baseline']['score']}, issues={len(report['baseline']['issues'])}",
        f"REFL: syntax_ok={report['reflected']['syntax_ok']}, score={report['reflected']['score']}, issues={len(report['reflected']['issues'])}",
        f"DELTA: score_gain={report['delta']['score_gain']} ({report['delta']['issues_before']} -> {report['delta']['issues_after']})",
        f"CHANGED_ONLY_SLOTS: {report['changed_only_slots']}",
        f"SANITIZED: {report['sanitized']}",
    ]
    _write(outdir / "02_langgraph_eval.txt", "\n".join(lines))

    _write(outdir / "ZZ_final_source.py", py_src_lg)
    okf, logf = _preflight(py_src_lg)
    _write(outdir / "ZZ_final_preflight.log", logf or "OK")
    print("[DONE] artifacts under:", outdir)


if __name__ == "__main__":
    main()
