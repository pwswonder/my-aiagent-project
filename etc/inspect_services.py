# scripts/inspect_services.py
# 목적: services 폴더의 파이썬 모듈을 동적으로 로드하지 않고(부작용 방지),
#       AST로 함수/클래스 시그니처를 추출해 콘솔에 요약 출력.
# 사용: python scripts/inspect_services.py

import os, ast, textwrap, inspect, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ROOT / "my_aiagent_project" / "services"


def parse_signatures(py_path: Path):
    try:
        source = py_path.read_text(encoding="utf-8")
    except Exception as e:
        return {"error": f"read_error: {e}"}
    try:
        tree = ast.parse(source)
    except Exception as e:
        return {"error": f"parse_error: {e}"}

    funcs, classes = [], []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args = []
            for a in node.args.args:
                args.append(a.arg)
            if node.args.vararg:
                args.append("*" + node.args.vararg.arg)
            if node.args.kwarg:
                args.append("**" + node.args.kwarg.arg)
            sig = f"{node.name}({', '.join(args)})"
            funcs.append(sig)
        elif isinstance(node, ast.ClassDef):
            bases = [getattr(b, "id", getattr(b, "attr", "")) for b in node.bases]
            classes.append({"name": node.name, "bases": bases})
    return {"funcs": funcs, "classes": classes}


def main():
    if not SERVICES.exists():
        print(f"[ERR] services dir not found: {SERVICES}")
        sys.exit(1)

    print(f"[INFO] scanning: {SERVICES}\n")
    for py in sorted(SERVICES.glob("*.py")):
        if py.name.startswith("_"):
            continue
        info = parse_signatures(py)
        print(f"--- {py.name} ---")
        if "error" in info:
            print("  ", info["error"])
            continue
        if info["classes"]:
            print("  classes:")
            for c in info["classes"]:
                print(f"    - {c['name']}({', '.join(c['bases'])})")
        if info["funcs"]:
            print("  functions:")
            for f in info["funcs"]:
                print(f"    - {f}")
        print()


if __name__ == "__main__":
    main()
