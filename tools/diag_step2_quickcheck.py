# tools/diag_step2_quickcheck.py
# -*- coding: utf-8 -*-
"""
생성된 최신 basecode에서 compile 블록과 변수 정의(optimizer/loss_fn/metrics)를 점검.
"""

import os, glob, json, re
from pathlib import Path

SEARCH_GLOBS = [
    "uploaded_docs/**/transformer_mt_basecode.py",
    "uploaded_docs/**/transformer_basecode.py",
]


def _latest(paths):
    cand = []
    for pat in paths:
        for f in glob.glob(pat, recursive=True):
            try:
                cand.append((os.path.getmtime(f), f))
            except OSError:
                pass
    if not cand:
        return None
    cand.sort(reverse=True)
    return cand[0][1]


def snippet(text, pos, span=300):
    lo = max(0, pos - span)
    hi = min(len(text), pos + span)
    return text[lo:hi]


def main():
    latest = _latest(SEARCH_GLOBS)
    if not latest:
        print("❌ basecode 파일을 찾지 못했습니다. 먼저 테스트를 실행하세요.")
        return
    print("[artifact]", latest)
    src = Path(latest).read_text(encoding="utf-8")

    # 1) 변수 정의 존재 여부
    has_opt = bool(re.search(r"(?m)^\s*optimizer\s*=", src))
    has_loss = bool(re.search(r"(?m)^\s*loss_fn\s*=", src))
    has_metrics = bool(re.search(r"(?m)^\s*metrics\s*=", src))
    print(f"[vars] optimizer={has_opt}, loss_fn={has_loss}, metrics={has_metrics}")

    # 2) 컴파일 호출 위치 및 주변
    m = re.search(r"model\.compile\s*\(", src)
    if not m:
        print("❌ model.compile(...) 호출이 없습니다.")
    else:
        print(f"[compile] found at index={m.start()}")
        print("----- compile vicinity -----")
        print(snippet(src, m.start()))
        print("----- end vicinity -----")

    # 3) 슬롯 주입 마커 잔존 여부(주입 실패 감지)
    leftovers = re.findall(r"CUSTOM_BLOCK:\s*([A-Za-z0-9_\-]+)", src)
    if leftovers:
        print("[WARN] 남아 있는 CUSTOM_BLOCK 마커:", leftovers)


if __name__ == "__main__":
    main()
