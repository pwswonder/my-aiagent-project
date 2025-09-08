# hotfix_slot_regex_surgical_v2.py
# -*- coding: utf-8 -*-
"""
services/codegen.py 안의 `_SLOT_LINE_RE = re.compile(...)`를
raw/비-raw 겸용 패턴으로 '안전 치환'합니다.
- 함수형 치환(lambda)로 역슬래시 이슈(bad escape \s) 방지
- 한 줄 끝에 붙은 쓰레기 조각까지 함께 제거
- .bak 백업 자동 생성
"""

from __future__ import annotations
from pathlib import Path
import re, sys

# 겸용 패턴(줄 주석 + {% raw %}/비-raw 모두 허용)
NEW_SLOT_LINE_RE = (
    r"(?m)^(?P<indent>[ \t]*)#\s*"
    r"(?:\{\%\s*raw\s*\%\}\s*)?"  # optional {% raw %}
    r"\{\{CUSTOM_BLOCK:(?P<name>[A-Za-z0-9_\-]+)\}\}"
    r"(?:\s*\{\%\s*endraw\s*\%\})?"  # optional {% endraw %}
    r"(?P<trail>[^\n]*)?$"
)

CODEGEN = Path("services/codegen.py")


def main():
    if not CODEGEN.exists():
        print(f"[ERR] not found: {CODEGEN.resolve()}", file=sys.stderr)
        sys.exit(1)

    src = CODEGEN.read_text(encoding="utf-8")

    # DOTALL+MULTILINE: 할당 시작줄부터 라인 끝까지 캡처해서 '라인 말' 쓰레기까지 제거
    # 그룹1: '_SLOT_LINE_RE = re.compile('까지
    # 그룹2: 컴파일 내부 패턴 본문(여러 줄 가능, 최소 탐욕)
    # 그룹3: 닫는 괄호 ')'
    # 그룹4: 같은 줄 끝까지의 잔여 쓰레기(있으면 제거)
    assign_rx = re.compile(
        r"(?ms)^(\s*_SLOT_LINE_RE\s*=\s*re\.compile\()"
        r"(.*?)"  # group 2: body
        r"(\))"  # group 3: closing paren
        r"([^\n]*)$"  # group 4: trailing junk on the same line
    )

    m = assign_rx.search(src)
    if not m:
        print(
            "[ERR] _SLOT_LINE_RE assignment not found in services/codegen.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # 함수형 치환: 역슬래시/정규식 메타를 있는 그대로 보존
    def repl(match: re.Match) -> str:
        prefix = match.group(1)  # '_SLOT_LINE_RE = re.compile('
        # group 2/3/4는 버리고 새 패턴으로 통째로 대체 (라인말 쓰레기도 제거)
        return f"{prefix}r'''{NEW_SLOT_LINE_RE}''')"

    new_src = assign_rx.sub(repl, src, count=1)
    if new_src == src:
        print("[ERR] substitution produced no change", file=sys.stderr)
        sys.exit(1)

    # 백업 + 쓰기
    CODEGEN.with_suffix(".py.bak").write_text(src, encoding="utf-8")
    CODEGEN.write_text(new_src, encoding="utf-8")
    print("[OK] Patched services/codegen.py (_SLOT_LINE_RE)")

    # 빠른 검증: import 가능 여부
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("services.codegen", CODEGEN)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "_SLOT_LINE_RE")
        print("[OK] Quick import test: PASS")
    except Exception as e:
        print("[WARN] Quick import test failed:", e, file=sys.stderr)


if __name__ == "__main__":
    main()
