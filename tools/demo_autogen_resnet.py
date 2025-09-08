

import os, sys, importlib.util, pathlib

ROOT = os.path.dirname(os.path.dirname(__file__))
tpl_candidates = [
    os.path.join(ROOT, "services", "templates", "resnet.slot.v1.j2"),
    os.path.join(ROOT, "resnet.slot.v1.j2"),
]
tpl = next((p for p in tpl_candidates if os.path.exists(p)), None)
if not tpl:
    print("[ERROR] resnet.slot.v1.j2 not found."); sys.exit(1)

spec = {
    "dims": {
        "stages": [
            {"filters": 64, "blocks": 2, "stride": 1},
            {"filters": 128, "blocks": 2, "stride": 2},
            {"filters": 256, "blocks": 2, "stride": 2},
            {"filters": 512, "blocks": 2, "stride": 2},
        ]
    }
}

ab_path = os.path.join(ROOT, "services", "codegen_autoblocks.py")
ab_spec = importlib.util.spec_from_file_location("codegen_autoblocks", ab_path)
ab = importlib.util.module_from_spec(ab_spec); sys.modules["codegen_autoblocks"] = ab; ab_spec.loader.exec_module(ab)
spec = ab.autofill_custom_blocks(spec, family="resnet")

inj_path = os.path.join(ROOT, "tools", "smoke_custom_block_test.py")
inj_spec = importlib.util.spec_from_file_location("smoke_custom_block_test", inj_path)
inj = importlib.util.module_from_spec(inj_spec); sys.modules["smoke_custom_block_test"] = inj; inj_spec.loader.exec_module(inj)

with open(tpl, "r", encoding="utf-8") as f:
    tpl_txt = f.read()

out_txt = inj.inject_custom_blocks(tpl_txt, spec.get("custom_blocks", {}))
out_path = os.path.join(ROOT, ".generated", "resnet_autogen.py")
pathlib.Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(out_txt)

print("✅ ResNet auto-gen completed")
print("→ Output:", out_path)
