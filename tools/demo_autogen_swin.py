import os, sys, importlib.util, pathlib
ROOT = os.path.dirname(os.path.dirname(__file__))
tpl = os.path.join(ROOT, "services", "templates", "swin.slot.v1.j2")

spec = {
    "family": "swin",
    "dims": {
        "swin": {
            "depths": [2, 2, 2, 2],
            "embed_dims": [64, 128, 256, 512],
            "window_size": 7,
            "mlp_ratio": 4.0,
            "patch_merging": True
        }
    }
}

ab_path = os.path.join(ROOT, "services", "codegen_autoblocks.py")
ab_spec = importlib.util.spec_from_file_location("codegen_autoblocks", ab_path)
ab = importlib.util.module_from_spec(ab_spec); sys.modules["codegen_autoblocks"] = ab; ab_spec.loader.exec_module(ab)
spec = ab.autofill_custom_blocks(spec, family="swin")

inj_path = os.path.join(ROOT, "tools", "smoke_custom_block_test.py")
inj_spec = importlib.util.spec_from_file_location("smoke_custom_block_test", inj_path)
inj = importlib.util.module_from_spec(inj_spec); sys.modules["smoke_custom_block_test"] = inj; inj_spec.loader.exec_module(inj)

with open(tpl, "r", encoding="utf-8") as f:
    txt = f.read()
out_txt = inj.inject_custom_blocks(txt, spec.get("custom_blocks", {}))
out_path = os.path.join(ROOT, ".generated", "swin_autogen.py")
pathlib.Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(out_txt)
print("✅ Swin-like backbone auto-gen ->", out_path)
