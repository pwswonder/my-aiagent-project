# llm_slot_re_hard_clean.py
# -*- coding: utf-8 -*-
"""
services/llm_codegen_assist.py 의
  SLOT_RE = re.compile( ... )
호출 전체(여러 줄 + 플래그/쉼표 포함)를 '겸용 패턴' 단일라인으로 교체하고,
호출 직후 이어지는 고아 플래그/쉼표 라인까지 제거합니다.
"""

from __future__ import annotations
from pathlib import Path
import re, sys, importlib.util

TARGET = Path("services/llm_codegen_assist.py")

NEW_SLOT_RE = (
    r"(?m)^\s*#\s*"
    r"(?:\{\%\s*raw\s*\%\}\s*)?"
    r"\{\{CUSTOM_BLOCK:([A-Za-z0-9_\-]+)\}\}"
    r"(?:\s*\{\%\s*endraw\s*\%\})?\s*$"
)


def find_span(text: str, name="SLOT_RE"):
    m = re.search(rf"(?m)^[ \t]*{name}\s*=\s*re\.compile\s*\(", text)
    if not m:
        return None
    i = m.end() - 1  # '(' 위치
    depth = 0
    in_str = False
    delim = ""
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if delim in ("'''", '"""'):
                if text.startswith(delim, i):
                    in_str = False
                    i += len(delim)
                    continue
                i += 1
                continue
            else:
                if ch == delim:
                    in_str = False
                i += 1
                continue
        else:
            if text.startswith("'''", i) or text.startswith('"""', i):
                delim = text[i : i + 3]
                in_str = True
                i += 3
                continue
            if ch in ("'", '"'):
                delim = ch
                in_str = True
                i += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return (m.start(), i + 1)
            i += 1
    return None


def main():
    if not TARGET.exists():
        print(f"[ERR] not found: {TARGET}", file=sys.stderr)
        sys.exit(1)
    text = TARGET.read_text(encoding="utf-8")
    span = find_span(text, "SLOT_RE")
    if not span:
        print("[ERR] SLOT_RE compile call not found", file=sys.stderr)
        sys.exit(1)

    lo, hi = span
    left, right = text[:lo], text[hi:]

    # 동일 들여쓰기 유지
    ls = left.rfind("\n") + 1
    indent = re.match(r"[ \t]*", left[ls:]).group(0)
    replacement = f"{indent}SLOT_RE = re.compile(r'''{NEW_SLOT_RE}''')\n"

    # 호출 직후의 '고아 플래그/쉼표' 라인 제거 (최대 30줄 스캔)
    lines = right.splitlines(keepends=True)
    cleaned = []
    taken = False
    for j, ln in enumerate(lines):
        # 빈 줄/주석은 그냥 지나감
        if re.match(r"^[ \t]*(#.*)?\n?$", ln):
            continue
        # 고아 플래그/쉼표 라인:   ',', 're.M', 're.S', 're.I', 're.X', 're.U' 등의 조합
        if re.match(r"^[ \t]*(,|re\.[A-Z_]+,?)\s*(#.*)?\n?$", ln):
            continue
        # 여기서부터는 정상 코드 → 나머지 붙이고 종료
        cleaned = lines[j:]
        taken = True
        break
    if not taken:
        cleaned = []

    new_text = left + replacement + "".join(cleaned)
    TARGET.with_suffix(".py.bak").write_text(text, encoding="utf-8")
    TARGET.write_text(new_text, encoding="utf-8")
    print("[OK] Clean replaced SLOT_RE in", TARGET)

    # 빠른 임포트 테스트
    try:
        spec = importlib.util.spec_from_file_location(
            "services.llm_codegen_assist", TARGET
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "SLOT_RE")
        print("[OK] Quick import test: PASS")
    except Exception as e:
        print("[WARN] Quick import test failed:", e, file=sys.stderr)


if __name__ == "__main__":
    main()
