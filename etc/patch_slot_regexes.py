# patch_slot_regexes.py
# -*- coding: utf-8 -*-
"""
- codegen.py, codegen_autoblocks.py, (선택) llm_codegen_assist.py의 CUSTOM_BLOCK 정규식을
  raw/비-raw 겸용 패턴으로 교체.
- 백업 파일(.bak) 자동 생성.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Tuple

# 통합 패턴(줄 주석 + raw/비-raw 모두 허용)
NEW_SLOT_LINE_RE = (
    r"(?m)^(?P<indent>[ \t]*)#\s*"
    r"(?:\{\%\s*raw\s*\%\}\s*)?"  # optional {% raw %}
    r"\{\{CUSTOM_BLOCK:(?P<name>[A-Za-z0-9_\-]+)\}\}"
    r"(?:\s*\{\%\s*endraw\s*\%\})?"  # optional {% endraw %}
    r"(?P<trail>[^\n]*)?$"
)

NEW_SLOT_RE_SIMPLE = (
    r"(?m)^\s*#\s*"
    r"(?:\{\%\s*raw\s*\%\}\s*)?"
    r"\{\{CUSTOM_BLOCK:([A-Za-z0-9_\-]+)\}\}"
    r"(?:\s*\{\%\s*endraw\s*\%\})?\s*$"
)


def patch_file(
    path: Path, assign_name: str, new_pattern_literal: str
) -> Tuple[bool, str]:
    """
    파일 내에서 `<assign_name> = re.compile(...)` 블록의 (...)을
    r'''<new_pattern_literal>''' 로 교체.
    - 함수형 치환(lambda)으로 역슬래시 이슈를 회피.
    """
    src = path.read_text(encoding="utf-8")

    # re.S로 줄바꿈 포함 매칭, body는 최소 탐욕으로 캡처
    rx = re.compile(
        rf"({re.escape(assign_name)}\s*=\s*re\.compile\()\s*(?P<body>.*?)\)", re.S
    )
    m = rx.search(src)
    if not m:
        return False, f"{assign_name} not found in {path.name}"

    def repl(match: re.Match) -> str:
        prefix = match.group(1)  # '<NAME> = re.compile(' 까지
        # r'''...''' 로 감싸 raw 문자열로 삽입 (백슬래시 그대로 보존)
        return f"{prefix}r'''{new_pattern_literal}''')"

    new_src = rx.sub(repl, src, count=1)
    if new_src == src:
        return False, f"{assign_name} unchanged (maybe already patched?)"

    # 백업 후 쓰기
    path.with_suffix(path.suffix + ".bak").write_text(src, encoding="utf-8")
    path.write_text(new_src, encoding="utf-8")
    return True, f"patched {assign_name} in {path.name}"


def main():
    root = Path(".").resolve()
    targets = [
        (root / "services/codegen.py", "_SLOT_LINE_RE", NEW_SLOT_LINE_RE),
        (root / "services/codegen_autoblocks.py", "SLOT_LINE_RE", NEW_SLOT_RE_SIMPLE),
        # 선택: LLM 경로도 겸용 패턴으로 통일하고 싶다면 아래 줄 유지
        (root / "services/llm_codegen_assist.py", "SLOT_RE", NEW_SLOT_RE_SIMPLE),
    ]
    for path, name, patt in targets:
        if not path.exists():
            print(f"[skip] {path.name} not found")
            continue
        ok, msg = patch_file(path, name, patt)
        print(("[OK] " if ok else "[INFO] ") + msg)


if __name__ == "__main__":
    main()
