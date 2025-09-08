
# -*- coding: utf-8 -*-
# Insert a safe FIT_KWARGS slot into *.j2 templates.
# Usage:
#   python tools/insert_fit_kwargs_slot.py services/templates
#   python tools/insert_fit_kwargs_slot.py services/templates/resnet.j2

import sys, re
from pathlib import Path

SLOT_LINE = '# {% raw %}{{CUSTOM_BLOCK:FIT_KWARGS}}{% endraw %}'

def find_closing_paren(text: str, start_idx: int) -> int:
    depth = 0
    i = start_idx
    while i < len(text):
        ch = text[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1

def insert_fit_kwargs(fp: Path) -> bool:
    s = fp.read_text(encoding='utf-8')
    if SLOT_LINE in s:
        return False  # already has slot
    m = re.search(r'\.fit\(', s)
    if not m:
        return False  # no fit in this template
    fit_start = m.start()
    pre = s[:fit_start]
    post = s[fit_start:]
    # indent of the line containing '.fit('
    lines_pre = pre.splitlines(True)
    fit_line = lines_pre[-1] if lines_pre else ""
    base_indent = fit_line[: len(fit_line) - len(fit_line.lstrip())]
    inner_indent = base_indent + (" " * 4)

    # pre-block (dict) above .fit(
    pre_block = (
        f"{base_indent}_fit_kwargs = {{\n"
        f"{inner_indent}{SLOT_LINE}\n"
        f"{base_indent}}}\n"
    )

    # closing ')' of this .fit(...)
    first_paren_rel = post.find('(')
    close_idx = find_closing_paren(post, first_paren_rel)
    if close_idx == -1:
        return False  # malformed; skip

    call_prefix = post[:close_idx]
    call_suffix = post[close_idx:]
    if not call_prefix.endswith('\n'):
        call_prefix += '\n'
    call_prefix += f"{inner_indent}**_fit_kwargs,\n"

    new_post = call_prefix + call_suffix
    new_text = pre + pre_block + new_post

    # backup + write
    bak = fp.with_suffix(fp.suffix + '.bak')
    if not bak.exists():
        bak.write_text(s, encoding='utf-8')
    fp.write_text(new_text, encoding='utf-8')
    return True

def main():
    if len(sys.argv) < 2:
        print('Usage: python tools/insert_fit_kwargs_slot.py <templates_dir_or_file.j2>')
        sys.exit(2)
    target = Path(sys.argv[1])
    if target.is_file() and target.suffix == '.j2':
        changed = insert_fit_kwargs(target)
        print(('FIXED ' if changed else 'OK    ') + str(target))
        return
    count = 0
    for fp in target.rglob('*.j2'):
        changed = insert_fit_kwargs(fp)
        print(('FIXED ' if changed else 'OK    ') + str(fp))
        if changed:
            count += 1
    print(f'Done. Files changed: {count}')

if __name__ == '__main__':
    main()
