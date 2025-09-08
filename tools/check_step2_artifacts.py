# tools/check_step2_artifacts.py
# -*- coding: utf-8 -*-
"""
Step 2 사후 점검:
1) 최근 생성된 basecode 파일 찾기
2) compile 인자 정렬 여부 확인
3) code_quality_analyzer.analyze_quality로 미스매치 없는지 확인
"""

import os
import sys
import json
import time
import glob
from pathlib import Path

# 성능 고려: glob를 좁게 (uploaded_docs/*/_basecode.py) 로 한정
SEARCH_GLOBS = [
    "uploaded_docs/**/transformer_mt_basecode.py",
    "uploaded_docs/**/transformer_basecode.py",
]


def _find_latest(paths):
    cand = []
    for p in paths:
        for f in glob.glob(p, recursive=True):
            try:
                cand.append((os.path.getmtime(f), f))
            except OSError:
                pass
    if not cand:
        return None
    cand.sort(reverse=True)
    return cand[0][1]


def main():
    # 1) 최신 산출물 찾기
    latest = _find_latest(SEARCH_GLOBS)
    if not latest:
        print(
            "❌ 최근 basecode 파일을 찾지 못했습니다. 테스트 스크립트를 먼저 실행하세요."
        )
        sys.exit(1)
    print(f"[artifact] {latest}")

    py_src = Path(latest).read_text(encoding="utf-8")
    # 2) compile 인자 정렬 확인
    must = "model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)"
    has_compile_align = must in py_src
    print(f"[check] compile aligned → {has_compile_align}")

    # 3) analyzer로 품질 확인
    #    (MT smoke 기준 spec: adam / sparse_categorical_crossentropy / accuracy)
    try:
        from services.code_quality_analyzer import analyze_quality
    except Exception as e:
        print("❌ analyzer import 실패:", e)
        sys.exit(1)

    spec = {
        "optimizer_name": "adam",
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"],
        # 필요시 entrypoint_name 고정 가능: "entrypoint_name": "build_model"
    }
    out = analyze_quality(py_src, spec)
    print("[analyze_quality] summary:", out.get("summary"))
    print(
        "[analyze_quality] issues:",
        json.dumps(out.get("issues", []), ensure_ascii=False),
    )

    # 4) 간단한 어서션
    issues = out.get("issues", [])
    mismatch_codes = {"OPT_MISMATCH", "LOSS_MISMATCH", "METRICS_MISMATCH"}
    has_mismatch = any(
        i.get("code") in mismatch_codes for i in issues if isinstance(i, dict)
    )
    if not has_compile_align or has_mismatch:
        print("❌ 정합성 불만족 (compile-align 또는 미스매치 존재)")
        sys.exit(2)

    print("✅ Step 2 post-check OK")


if __name__ == "__main__":
    main()
