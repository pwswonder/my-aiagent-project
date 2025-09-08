
"""
Scan a template for CUSTOM_BLOCK markers and list detected slots.
Handles both:
  - # {{CUSTOM_BLOCK:slot}} / # {CUSTOM_BLOCK:slot}
  - with optional {% raw %} ... {% endraw %}
Usage:
  python tools/scan_slots.py --template services/templates/transformer.slot.v1.j2
"""
import argparse, os, re, sys

SLOT_LINE = re.compile(
    r"^(?P<indent>\s*)\#\s*"
    r"(?:\{\%\s*raw\s*\%\}\s*)?"
    r"\{\{?\s*CUSTOM_BLOCK:(?P<slot>[^}\s]+)\s*\}\}?"
    r"(?:\s*\{\%\s*endraw\s*\%\})?"
    r"(?:\s*\#.*)?$",
    re.MULTILINE
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    args = ap.parse_args()
    path = args.template
    if not os.path.exists(path):
        print(f"[ERROR] Not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    slots = [m.group("slot") for m in SLOT_LINE.finditer(txt)]
    print("Detected slots:", sorted(set(slots)))

if __name__ == "__main__":
    main()
