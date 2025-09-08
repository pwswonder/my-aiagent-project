# -*- coding: utf-8 -*-
# Fix/normalize the compile_override slot line around `model.compile(...)`.
#
# What this does (idempotent):
# 1) If the slot and `model.compile(` are on the SAME line, split them into TWO lines.
# 2) Ensure the slot line indent MATCHES the indent of the following `model.compile(` line.
# 3) Leave everything else untouched. Creates a `.bak` backup on first change.
#
# Usage:
#   python tools/fix_compile_override_indent.py services/templates
#   python tools/fix_compile_override_indent.py services/templates/resnet.j2
import sys, re
from pathlib import Path

SLOT = "# {% raw %}{{CUSTOM_BLOCK:compile_override}}{% endraw %}"


def _leading_ws(s: str) -> str:
    return s[: len(s) - len(s.lstrip())]


def _split_slot_and_compile(line: str):
    """If the line contains both the slot and 'model.compile(', split into two lines
    using the indent just before 'model.compile(' for BOTH lines."""
    if SLOT not in line or "model.compile(" not in line:
        return None
    idx = line.find("model.compile(")
    compile_indent = _leading_ws(line[:idx])
    nl = "" if line.endswith("\n") else "\n"
    return f"{compile_indent}{SLOT}\n{compile_indent}{line[idx:].lstrip()}{nl}"


def fix_file(fp: Path) -> bool:
    s = fp.read_text(encoding="utf-8")
    lines = s.splitlines(True)
    changed = False
    # Pass 1: split one-line cases
    for i, ln in enumerate(lines):
        rep = _split_slot_and_compile(ln)
        if rep is not None and rep != ln:
            lines[i] = rep
            changed = True
    # Pass 2: align indent of pure slot line to the following compile line (if adjacent)
    i = 0
    while i < len(lines) - 1:
        ln = lines[i]
        nx = lines[i + 1]
        if ln.strip() == SLOT and "model.compile(" in nx:
            slot_ws = _leading_ws(ln)
            compile_ws = _leading_ws(nx)
            if slot_ws != compile_ws:
                nl = "" if ln.endswith("\n") else "\n"
                lines[i] = f"{compile_ws}{SLOT}{nl}"
                changed = True
        i += 1
    if changed:
        bak = fp.with_suffix(fp.suffix + ".bak")
        if not bak.exists():
            bak.write_text(s, encoding="utf-8")
        fp.write_text("".join(lines), encoding="utf-8")
    return changed


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python tools/fix_compile_override_indent.py <templates_dir_or_file.j2>"
        )
        sys.exit(2)
    target = Path(sys.argv[1])
    if target.is_file() and target.suffix == ".j2":
        print(("FIXED " if fix_file(target) else "OK    ") + str(target))
        return
    count = 0
    for fp in target.rglob("*.j2"):
        c = fix_file(fp)
        print(("FIXED " if c else "OK    ") + str(fp))
        count += int(c)
    print(f"Done. Fixed files: {count}")


if __name__ == "__main__":
    main()
