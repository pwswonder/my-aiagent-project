
from pathlib import Path
from services.lib.slot_payload_resolver import resolve_payloads_for_template

spec = {"loss": "sparse_categorical_crossentropy"}
tpl_dir = Path("services/templates")

for j2 in tpl_dir.rglob("*.j2"):
    text = j2.read_text(encoding="utf-8", errors="ignore")
    try:
        resolve_payloads_for_template(spec, text, j2.name)
        print(f"[OK] {j2}")
    except Exception as e:
        print(f"[FAIL] {j2}: {e}")
