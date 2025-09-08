# trace_reflection_regression.py
# -*- coding: utf-8 -*-
"""
특정 템플릿(e.g., rnn_seq)의 E2E 산출물을 스캔해
- 01_baseline_slot_applied.py
- 02_preseed_for_reflection.py
- 03_after_langgraph.py
- 04_after_quality_reflection.py
사이의 차이/변경 라인 수와 평가 리포트(02_langgraph_eval.json)를 요약한다.
"""

from __future__ import annotations
import json, difflib
from pathlib import Path


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def diff_stats(a: str, b: str):
    d = list(difflib.unified_diff(a.splitlines(), b.splitlines()))
    # 단순 지표: 변경 라인 수
    changes = sum(
        1 for x in d if x.startswith(("+", "-")) and not x.startswith(("+++", "---"))
    )
    return changes


def main():
    base = Path("e2e_eval_out/rnn_seq")
    f01 = base / "01_baseline_slot_applied.py"
    f02 = base / "02_preseed_for_reflection.py"
    f03 = base / "03_after_langgraph.py"
    f04 = base / "04_after_quality_reflection.py"
    fEval = base / "02_langgraph_eval.json"

    s01 = read(f01)
    s02 = read(f02)
    s03 = read(f03)
    s04 = read(f04)
    print("[files]")
    for p in [f01, f02, f03, f04]:
        print(
            " -",
            p.name,
            "exists:",
            p.exists(),
            "| size:",
            p.stat().st_size if p.exists() else 0,
        )

    print("\n[diff changes]")
    print(" 01→02:", diff_stats(s01, s02))
    print(" 02→03:", diff_stats(s02, s03))
    print(" 03→04:", diff_stats(s03, s04))

    if fEval.exists():
        ev = json.loads(fEval.read_text(encoding="utf-8"))
        print("\n[eval]")
        print(" score_gain:", ev.get("score_gain"))
        print(" issues_before:", ev.get("issues_before"))
        print(" issues_after:", ev.get("issues_after"))
        print(" notes:", ev.get("notes") or ev.get("report") or "")


if __name__ == "__main__":
    main()
