#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repo Introspector
- Scans a Python project (e.g., your AI 기술논문 agent repo) and produces:
  1) JSON manifest with functions/classes/imports per file
  2) Markdown report highlighting duplicates and the internal dependency graph
Usage:
  python repo_introspect.py --root /path/to/project --out /path/to/save
Notes:
  - Only .py files are analyzed.
  - Internal edges are based on module basenames or "services.<mod>" imports.
"""
import os, re, ast, json, argparse, textwrap
from typing import Dict, Any, List, Tuple, Set

def _parse_py(path: str) -> Dict[str, Any]:
    try:
        src = open(path, "r", encoding="utf-8").read()
    except Exception as e:
        return {"error": f"read-error: {e}"}
    try:
        tree = ast.parse(src)
    except Exception as e:
        return {"error": f"parse-error: {e}"}
    info: Dict[str, Any] = {
        "docstring": ast.get_docstring(tree) or "",
        "functions": [],
        "classes": [],
        "imports": []
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            info["functions"].append({
                "name": node.name,
                "lineno": node.lineno,
                "args": [a.arg for a in node.args.args]
            })
        elif isinstance(node, ast.ClassDef):
            methods = [b.name for b in node.body if isinstance(b, ast.FunctionDef)]
            info["classes"].append({
                "name": node.name,
                "lineno": node.lineno,
                "methods": methods
            })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                info["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            info["imports"].append(mod)
    return info

def _internal_edges(manifest: Dict[str, Any]) -> List[Tuple[str, str]]:
    module_names = {os.path.splitext(fn)[0]: fn for fn in manifest.keys()}
    edges: Set[Tuple[str, str]] = set()
    for fn, info in manifest.items():
        if not isinstance(info, dict): continue
        imports = info.get("imports", [])
        srcmod = os.path.splitext(fn)[0]
        for imp in imports:
            base = (imp or "").split(".")[-1]
            if base in module_names:
                edges.add((srcmod, base))
            elif (imp or "").startswith("services."):
                base2 = imp.split(".")[1]
                if base2 in module_names:
                    edges.add((srcmod, base2))
    return sorted(edges)

def _duplicates(manifest: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    func_map: Dict[str, List[str]] = {}
    class_map: Dict[str, List[str]] = {}
    for fn, info in manifest.items():
        if not isinstance(info, dict): continue
        for f in info.get("functions", []):
            func_map.setdefault(f["name"], []).append(fn)
        for c in info.get("classes", []):
            class_map.setdefault(c["name"], []).append(fn)
    dupe_funcs = {k: v for k, v in func_map.items() if len(v) > 1}
    dupe_classes = {k: v for k, v in class_map.items() if len(v) > 1}
    return {"functions": dupe_funcs, "classes": dupe_classes}

def _emit_markdown(manifest: Dict[str, Any], edges: List[Tuple[str, str]], dups: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Repo Introspection Report\n")
    lines.append("## 1) Files Scanned\n")
    for fn in sorted(manifest.keys()):
        lines.append(f"- `{fn}`")
    lines.append("\n## 2) Internal Dependency Graph (file -> imported-file)\n")
    if edges:
        for a, b in edges:
            lines.append(f"- `{a}` → `{b}`")
    else:
        lines.append("- (no internal edges found)")
    lines.append("\n## 3) Duplicates\n")
    lines.append("### Functions with the same name in multiple files\n")
    if dups["functions"]:
        for name, flist in sorted(dups["functions"].items()):
            lines.append(f"- **{name}**: " + ", ".join(f"`{f}`" for f in sorted(set(flist))))
    else:
        lines.append("- None")
    lines.append("\n### Classes with the same name in multiple files\n")
    if dups["classes"]:
        for name, flist in sorted(dups["classes"].items()):
            lines.append(f"- **{name}**: " + ", ".join(f"`{f}`" for f in sorted(set(flist))))
    else:
        lines.append("- None")
    lines.append("\n## 4) File Summaries\n")
    for fn in sorted(manifest.keys()):
        info = manifest[fn]
        if not isinstance(info, dict):
            lines.append(f"\n### {fn}\n- ERROR: {info}")
            continue
        lines.append(f"\n### {fn}")
        doc = (info.get("docstring") or "").strip().splitlines()
        if doc:
            lines.append("**Docstring (first lines):**")
            for ln in doc[:5]:
                lines.append(f"> {ln}")
        fs = ", ".join(f["name"] for f in info.get("functions", []))
        cs = ", ".join(c["name"] for c in info.get("classes", []))
        lines.append(f"- Functions: {fs or '(none)'}")
        lines.append(f"- Classes: {cs or '(none)'}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Repo root directory to scan")
    ap.add_argument("--out", required=True, help="Directory to write outputs")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    manifest: Dict[str, Any] = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".py"): continue
            if any(seg in dirpath for seg in (".git", ".venv", "venv", "__pycache__")):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            manifest[rel] = _parse_py(full)

    edges = _internal_edges(manifest)
    dups = _duplicates(manifest)

    json_out = os.path.join(out, "repo_manifest.json")
    md_out = os.path.join(out, "repo_report.md")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "file_count": len(manifest),
                "edge_count": len(edges),
                "dupe_func_count": len(dups["functions"]),
                "dupe_class_count": len(dups["classes"]),
            },
            "manifest": manifest,
            "edges": edges,
            "duplicates": dups
        }, f, ensure_ascii=False, indent=2)
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(_emit_markdown(manifest, edges, dups))

    print("Wrote:", json_out)
    print("Wrote:", md_out)

if __name__ == "__main__":
    main()
