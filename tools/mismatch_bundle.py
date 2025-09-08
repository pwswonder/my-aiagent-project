# tools/mismatch_bundle.py
# -*- coding: utf-8 -*-
"""
요청 파일들을 점검하고 존재하는 것만 zip으로 묶습니다.
- 실행: python tools/mismatch_bundle.py
- 산출: mt_debug_bundle.zip (같은 디렉토리)
"""

import os, sys, zipfile

REQUESTED = [
    "services/template_registry.py",
    "services/codegen.py",
    "services/code_quality_analyzer.py",
    "services/quality_reflection.py",
    "services/llm_codegen_assist.py",
    "services/codegen_autoblocks.py",
    "services/spec_hardener.py",
    "services/routing.py",
    # 템플릿 경로는 프로젝트에 따라 다를 수 있음: 아래 경로 중 존재하는 것만 포함
    "services/templates/transformer_mt.j2",
    "templates/transformer_mt.j2",
    # 생성 결과(경로는 로그에 나온 실제 경로로 교체해 주세요)
    "uploaded_docs/doc_step1_mt_smoke/transformer_mt_basecode.py",
]

def main():
    ok, miss = [], []
    for p in REQUESTED:
        if os.path.exists(p):
            ok.append(p)
        else:
            miss.append(p)

    print("[FOUND]")
    for p in ok:
        print(" -", p)
    print("\n[MISSING]")
    for p in miss:
        print(" -", p)

    if not ok:
        print("\n❌ Nothing to zip. Adjust paths in REQUESTED.")
        sys.exit(1)

    out_zip = "mt_debug_bundle.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in ok:
            zf.write(p, arcname=p)
    print(f"\n✅ Wrote: {out_zip} (attach this file here)")

if __name__ == "__main__":
    main()
