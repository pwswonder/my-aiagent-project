# -*- coding: utf-8 -*-
# Rename **_fit_kwargs to **FIT_KWARGS and the dict name accordingly in *.j2 templates.

import sys, re
from pathlib import Path

def patch_file(fp: Path) -> bool:
    s = fp.read_text(encoding='utf-8')
    orig = s
    # Replace the dict name if present: _fit_kwargs = { ... } -> FIT_KWARGS = { ... }
    s = re.sub(r'(^[ \t]*)_fit_kwargs\s*=\s*\{', r'\1FIT_KWARGS = {', s, flags=re.M)
    # Replace usage in fit call: **_fit_kwargs -> **FIT_KWARGS
    s = s.replace('**_fit_kwargs', '**FIT_KWARGS')
    if s != orig:
        fp.write_text(s, encoding='utf-8')
        return True
    return False

def main():
    if len(sys.argv) < 2:
        print('Usage: python tools/rename_fit_kwargs_arg.py <templates_dir_or_file.j2>')
        raise SystemExit(2)
    target = Path(sys.argv[1])
    if target.is_file() and target.suffix == '.j2':
        print(('FIXED ' if patch_file(target) else 'OK    ') + str(target))
        return
    count = 0
    for fp in target.rglob('*.j2'):
        if patch_file(fp):
            print('FIXED', fp)
            count += 1
        else:
            print('OK   ', fp)
    print('Done. Files changed:', count)

if __name__ == '__main__':
    main()
