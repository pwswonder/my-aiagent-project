# -*- coding: utf-8 -*-
"""
Hotfix: Ensure the generated builder function returns a model.

Usage:
  python tools/patch_generated_return.py --file .generated/resnet_generated.py --func build_model --var model

- Finds `def <func>(...)` and checks whether a `return` exists.
- If missing, it tries to insert `return <var>` just before the function dedents.
- Creates a backup: <file>.bak
- Prints what it did.

Heuristics:
- Default var candidates: model, net, generator, autoencoder, g_model
- Indentation is inferred from the first body line.
"""
import argparse, re
from pathlib import Path

def find_function_block(lines, func_name):
    pattern = re.compile(r'^([ \t]*)def\s+' + re.escape(func_name) + r'\s*\(.*\)\s*:\s*(?:#.*)?$', re.M)
    for i, ln in enumerate(lines):
        m = pattern.match(ln)
        if m:
            def_indent = m.group(1)
            start = i
            # find body start
            j = i + 1
            while j < len(lines) and (lines[j].strip() == '' or lines[j].lstrip().startswith('#')):
                j += 1
            if j >= len(lines):
                return def_indent, start, j, j
            body_indent = re.match(r'^([ \t]*)', lines[j]).group(1)
            # walk to end where indent <= def_indent (function dedent)
            k = j
            while k < len(lines):
                ind = re.match(r'^([ \t]*)', lines[k]).group(1)
                if lines[k].strip() and len(ind) <= len(def_indent) and k > j:
                    break
                k += 1
            end = k  # exclusive
            return def_indent, start, j, end
    return None, -1, -1, -1

def has_return(lines, body_start, body_end):
    for ln in lines[body_start:body_end]:
        if ln.strip().startswith('return '):
            return True
    return False

def pick_var(lines, body_start, body_end, preferred):
    # preferred first
    if preferred:
        return preferred
    # heuristics
    candidates = ['model', 'net', 'generator', 'autoencoder', 'g_model']
    for name in candidates:
        # quick scan for assignment like "name = " inside the function
        for ln in lines[body_start:body_end]:
            if re.match(r'^\s*' + re.escape(name) + r'\s*=', ln):
                return name
    # fallback
    return 'model'

def patch_file(path, func_name, var_name=None):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    txt = p.read_text(encoding='utf-8')
    lines = txt.splitlines(True)

    def_indent, start, body_start, body_end = find_function_block(lines, func_name)
    if start < 0:
        print(f"[patch] function '{func_name}' not found in {path}")
        return False

    if has_return(lines, body_start, body_end):
        print(f"[patch] function '{func_name}' already has a return; no change.")
        return True

    # determine body indent level
    if body_start < len(lines):
        body_indent = re.match(r'^([ \t]*)', lines[body_start]).group(1)
        if len(body_indent) <= len(def_indent):
            body_indent = def_indent + '    '
    else:
        body_indent = def_indent + '    '

    var = pick_var(lines, body_start, body_end, var_name)

    insert_at = body_end
    # insert before dedent
    lines.insert(insert_at, f"{body_indent}return {var}\n")

    # backup and write
    bak = p.with_suffix(p.suffix + ".bak")
    bak.write_text(txt, encoding='utf-8')
    p.write_text("".join(lines), encoding='utf-8')
    print(f"[patch] inserted 'return {var}' into '{func_name}' in {path}")
    print(f"[patch] backup saved at {bak}")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--func", default="build_model")
    ap.add_argument("--var", default=None)
    args = ap.parse_args()

    ok = patch_file(args.file, args.func, args.var)
    if not ok:
        raise SystemExit(2)

if __name__ == "__main__":
    main()