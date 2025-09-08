
"""
Insert **FIT_KWARGS into model.fit(...) calls if missing.
- Makes a backup file with .bak extension next to each modified file.
- Safe heuristic regex; review diffs.

Usage:
  python tools/auto_insert_fit_kwargs.py TEMPLATE_DIR
"""
from pathlib import Path
import re, sys, shutil

FIT_RE = re.compile(r"(model\.fit\s*\()([^\)]*)(\))", re.S)

def insert_kwargs_in_text(text: str) -> str:
    def _repl(m):
        head, args, tail = m.group(1), m.group(2), m.group(3)
        if "**FIT_KWARGS" in args:
            return m.group(0)  # already present
        args = args.rstrip()
        if args and args[-1] != ",":
            args = args + ", "
        args = args + "**FIT_KWARGS"
        return head + args + tail
    # Replace only the first occurrence; run multiple times if needed
    return FIT_RE.sub(_repl, text, count=1)

def process_dir(d: Path):
    for p in d.glob("*.j2"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "model.fit" not in text:
            continue
        if "**FIT_KWARGS" in text:
            continue
        new_text = insert_kwargs_in_text(text)
        if new_text != text:
            shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
            p.write_text(new_text, encoding="utf-8")
            print("Patched:", p.name)

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/auto_insert_fit_kwargs.py TEMPLATE_DIR")
        sys.exit(1)
    d = Path(sys.argv[1])
    if not d.exists():
        print("Directory not found:", d)
        sys.exit(1)
    process_dir(d)

if __name__ == "__main__":
    main()
