
# -*- coding: utf-8 -*-
from __future__ import annotations
import difflib, re, sys
from pathlib import Path
from typing import Dict, List

SLOT_RE = re.compile(r"^([ \t]*)#\s*\{\%\s*raw\s*\%\}\{\{CUSTOM_BLOCK:([a-zA-Z0-9_]+)\}\}\{\%\s*endraw\s*\%\}\s*$", re.M)

def _find_slot_lines(src: str) -> Dict[int, str]:
    mapping = {}
    for i, ln in enumerate(src.splitlines(), start=1):
        if SLOT_RE.match(ln):
            mapping[i] = "slot"
    return mapping

def _diff_changed_lines(a: str, b: str) -> List[int]:
    sm = difflib.SequenceMatcher(None, a.splitlines(True), b.splitlines(True))
    changed = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace","delete"):
            changed.extend(range(i1+1, i2+1))  # 1-based
    return changed

def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/inspect_non_slot_changes.py <baseline.py> <reflected.py>")
        sys.exit(1)
    base = Path(sys.argv[1]).read_text(encoding="utf-8")
    refl = Path(sys.argv[2]).read_text(encoding="utf-8")
    slots = _find_slot_lines(base)
    changed = _diff_changed_lines(base, refl)
    offenders = [ln for ln in changed if ln not in slots]
    if not offenders:
        print("No non-slot baseline lines were modified/deleted.")
        return
    print(f"Changed baseline lines outside slots: {len(offenders)}")
    base_lines = base.splitlines()
    for ln in offenders[:100]:
        s = base_lines[ln-1] if 1 <= ln <= len(base_lines) else ""
        print(f"{ln:5d}: {s}")

if __name__ == "__main__":
    main()
