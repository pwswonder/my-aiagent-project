
# -*- coding: utf-8 -*-
# tools/align_any_slot_indent.py
#
# Purpose:
# - Fix IndentationError caused by CUSTOM_BLOCK slot lines with wrong indentation.
# - Splits 'slot + code' on the same line into two lines.
# - Aligns the slot line's indentation to match the next real code line.
#
# Usage:
#   python tools/align_any_slot_indent.py services/templates --slot compile_override
#   python tools/align_any_slot_indent.py services/templates/gan.j2 --slot compile_override

import sys, re
from pathlib import Path
from typing import Optional

# Match both raw-guarded and plain slot markers on one line; capture trailing text as 'rest'.
SLOT_LINE_RE = re.compile(
    r'^([ \t]*)#\s*(?:\{\%\s*raw\s*\%\}\s*)?\{\{CUSTOM_BLOCK:([A-Za-z0-9_]+)\}\}(?:\s*\{\%\s*endraw\s*\%\})?\s*(.*)$'
)

def leading_ws(s: str) -> str:
    return s[: len(s) - len(s.lstrip())]

def is_comment_or_empty(s: str) -> bool:
    st = s.strip()
    return (not st) or st.startswith('#')

def align_file(fp: Path, target_slot: Optional[str]) -> bool:
    s = fp.read_text(encoding='utf-8')
    lines = s.splitlines(True)
    changed = False

    i = 0
    while i < len(lines):
        ln = lines[i]
        m = SLOT_LINE_RE.match(ln)
        if not m:
            i += 1
            continue

        indent, slot, rest = m.groups()
        if target_slot and slot != target_slot:
            i += 1
            continue

        # Find the indent of the next real code line (non-empty, non-comment)
        j = i + 1
        while j < len(lines) and is_comment_or_empty(lines[j]):
            j += 1
        next_indent = leading_ws(lines[j]) if j < len(lines) else indent

        # Prepare canonical slot marker line (avoid f-string with Jinja braces)
        slot_marker = "{% raw %}" + "{{CUSTOM_BLOCK:" + slot + "}}" + "{% endraw %}"
        new_slot_line = next_indent + "# " + slot_marker + "\n"

        # 1) If 'slot + code' on the same line, split them
        if rest.strip():
            # replace current line with slot only
            lines[i] = new_slot_line
            # insert the 'rest' as a separate line, aligned with next_indent
            rest_line = next_indent + rest.lstrip()
            if not rest_line.endswith("\n"):
                rest_line += "\n"
            lines.insert(i + 1, rest_line)
            changed = True
            i += 2
            continue
        else:
            # ensure slot line indent matches next code line
            if ln != new_slot_line:
                lines[i] = new_slot_line
                changed = True
                i += 1
                continue

        i += 1

    if changed:
        bak = fp.with_suffix(fp.suffix + '.bak')
        if not bak.exists():
            bak.write_text(s, encoding='utf-8')
        fp.write_text(''.join(lines), encoding='utf-8')
    return changed

def main():
    if len(sys.argv) < 2:
        print('Usage: python tools/align_any_slot_indent.py <templates_dir_or_file.j2> [--slot <name>]')
        sys.exit(2)

    target = Path(sys.argv[1])
    slot = None
    if '--slot' in sys.argv:
        idx = sys.argv.index('--slot')
        if idx + 1 < len(sys.argv):
            slot = sys.argv[idx + 1]

    if target.is_file() and target.suffix == '.j2':
        print(('FIXED ' if align_file(target, slot) else 'OK    ') + str(target))
        return

    if target.is_dir():
        count = 0
        for fp in target.rglob('*.j2'):
            if align_file(fp, slot):
                print('FIXED', fp)
                count += 1
            else:
                print('OK   ', fp)
        print(f'Done. Files changed: {count}')
    else:
        print('Target must be a directory or a .j2 file.')

if __name__ == '__main__':
    main()
