# patch_e2e_analyzer_call.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import re, sys


def patch_runfile(path: Path):
    code = path.read_text(encoding="utf-8")
    # 매우 보수적인 패치: code_path=... 패턴을 code=Path(...).read_text(...)으로 치환
    code_new = re.sub(
        r"analyze_quality\s*\(\s*code_path\s*=\s*([^)]+)\)",
        r"analyze_quality(code=Path(\1).read_text(encoding='utf-8'))",
        code,
    )
    if code_new != code and "from pathlib import Path" not in code_new:
        code_new = "from pathlib import Path\n" + code_new
    path.write_text(code_new, encoding="utf-8")
    print("[OK] patched:", path)


if __name__ == "__main__":
    run_path = Path("run_e2e_reflection_eval.py")
    if not run_path.exists():
        print("run_e2e_reflection_eval.py not found in current dir", file=sys.stderr)
        sys.exit(1)
    patch_runfile(run_path)
