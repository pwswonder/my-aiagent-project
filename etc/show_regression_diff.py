# show_regression_diff.py
# -*- coding: utf-8 -*-
"""
rnn_seq의 02→03 변경 라인을 문맥과 함께 보여준다.
- 변경 키워드(optimizer/loss/metrics/return/shape 등) 하이라이트 표시
"""
from __future__ import annotations
import difflib, re, json
from pathlib import Path

TEMPLATE = "rnn_seq"  # 필요시 변경


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


HL = re.compile(
    r"(optimizer|loss|metrics|activation|return|shape|mask|compile|LSTM|GRU|Dropout|BatchNorm|dtype)"
)


def highlight(line: str) -> str:
    return HL.sub(lambda m: f"<<{m.group(1)}>>", line)


def main():
    base = Path("e2e_eval_out") / TEMPLATE
    f02, f03, feval = (
        base / "02_preseed_for_reflection.py",
        base / "03_after_langgraph.py",
        base / "02_langgraph_eval.json",
    )
    s02, s03 = read(f02), read(f03)
    diff = list(
        difflib.unified_diff(
            s02.splitlines(),
            s03.splitlines(),
            fromfile="02_preseed",
            tofile="03_after_langgraph",
            lineterm="",
        )
    )
    print("# --- DIFF (02 -> 03) ---")
    for line in diff:
        if line.startswith(("+", "-", "@@")):
            print(highlight(line))
    if feval.exists():
        try:
            ev = json.loads(feval.read_text(encoding="utf-8"))
            print("\n# --- Eval summary ---")
            print(
                "score_gain:",
                ev.get("score_gain"),
                "| issues_before:",
                ev.get("issues_before"),
                "| issues_after:",
                ev.get("issues_after"),
            )
            notes = ev.get("notes") or ev.get("report") or ""
            if notes:
                print("notes:", notes[:1000])
        except Exception as e:
            print("eval read error:", e)


if __name__ == "__main__":
    main()
