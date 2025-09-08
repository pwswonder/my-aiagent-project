# audit_and_scaffold_templates.py
# -*- coding: utf-8 -*-
"""
- services/templates/**/*.j2 스캔
- 파일명 힌트로 권장 슬롯 추론(vision/rnn/transformer 등)
- 누락 슬롯을 파일 끝에 안전하게 placeholder로 스캐폴드(복사본만 생성)
- coverage_report.csv 생성
"""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(".")  # 리포 루트
BLOCK_RE = re.compile(r"\{\{CUSTOM_BLOCK:([A-Za-z0-9_]+)\}\}")

ESSENTIAL = ["imports_extra","compile_override","FIT_KWARGS","callbacks"]
STRUCTURE = ["head","stages","inception_mixed","rnn_stack","decoder_layers","model_body_extra"]

def infer_recommended(name: str):
    n = name.lower()
    rec = set(ESSENTIAL + ["head"])
    if any(k in n for k in ["inception","googlenet"]):
        rec |= {"inception_mixed","model_body_extra"}
    if any(k in n for k in ["cnn","resnet","vgg","conv","vision","image"]):
        rec |= {"stages","model_body_extra"}
    if any(k in n for k in ["rnn","lstm","gru","seq","sequence","timeseries"]):
        rec |= {"rnn_stack","model_body_extra"}
    if any(k in n for k in ["transformer","decoder","bert","gpt","translation","seq2seq"]):
        rec |= {"decoder_layers","model_body_extra"}
    return sorted(rec)

def insert_placeholders(src: str, missing: list) -> str:
    # 원본 안전: 파일 끝에만 추가
    lines = src.splitlines()
    lines += ["", "# === AUTO-SCAFFOLDED CUSTOM BLOCKS (safe to move) ==="]
    for blk in missing:
        lines.append(f"# {{% raw %}}{{{{CUSTOM_BLOCK:{blk}}}}}{{% endraw %}}")
    return "\n".join(lines) + "\n"

def main():
    j2_files = sorted(Path("services/templates").rglob("*.j2"))
    rows = []
    out_root = Path("templates_scaffolded")
    out_root.mkdir(parents=True, exist_ok=True)

    for j2 in j2_files:
        text = j2.read_text(encoding="utf-8", errors="ignore")
        found = sorted(set(BLOCK_RE.findall(text)))
        rec = infer_recommended(j2.name)
        missing = [b for b in rec if b not in found]
        coverage = round(len(found) / max(1, len(rec)), 2)
        rows.append({
            "template": str(j2),
            "found_blocks": ", ".join(found) if found else "(none)",
            "recommended_blocks": ", ".join(rec),
            "missing_blocks": ", ".join(missing) if missing else "",
            "coverage_ratio": coverage,
        })
        if missing:
            out_path = out_root / j2
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(insert_placeholders(text, missing), encoding="utf-8")

    pd.DataFrame(rows).sort_values(["coverage_ratio","template"]).to_csv(
        "coverage_report.csv", index=False, encoding="utf-8"
    )

if __name__ == "__main__":
    main()
