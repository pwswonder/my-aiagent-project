"""
Scan .j2 templates and report:
  - has model.fit(...)?
  - contains **FIT_KWARGS in fit call?
  - has FIT_KWARGS slot at top?

Usage:
  python tools/check_fit_kwargs_usage.py [TEMPLATE_DIR ...]
  # If no args given, uses /mnt/data/with_llm_slots and /mnt/data/with_llm_slots_fixed if present.
"""

from pathlib import Path
import re, sys


def analyze_file(p: Path):
    text = p.read_text(encoding="utf-8", errors="ignore")
    has_fit = bool(re.search(r"\bmodel\.fit\s*\(", text))
    uses_kwargs = bool(
        re.search(r"\bmodel\.fit\s*\([^)]*\*\*FIT_KWARGS", text, flags=re.S)
    )
    has_slot = ("CUSTOM_BLOCK:FIT_KWARGS" in text) or ("FIT_KWARGS" in text)
    return has_fit, uses_kwargs, has_slot


def main():
    args = sys.argv[1:] or []
    dirs = []
    if not args:
        for d in ["./services/templates/"]:
            dp = Path(d)
            if dp.exists():
                dirs.append(dp)
    else:
        dirs = [Path(a) for a in args]

    targets = []
    for d in dirs:
        for p in d.glob("*.j2"):
            targets.append(p)
    if not targets:
        print("No .j2 files found in:", [str(d) for d in dirs])
        sys.exit(0)

    print(f"{'file':40}  {'fit()':6}  {'**FIT_KWARGS':12}  {'slot':6}")
    print("-" * 70)
    for p in sorted(targets):
        has_fit, uses_kwargs, has_slot = analyze_file(p)
        print(
            f"{p.name:40}  {str(has_fit):6}  {str(uses_kwargs):12}  {str(has_slot):6}"
        )


if __name__ == "__main__":
    main()
