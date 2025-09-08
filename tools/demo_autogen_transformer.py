
"""
Demo: Structured spec -> auto-generated slot code (Transformer encoder_layers)
-----------------------------------------------------------------------------
1) We create a minimal structured spec with dims.encoder_layers.count.
2) We call services.codegen_autoblocks.autofill_custom_blocks(...) to synthesize code.
3) We reuse tools/smoke_custom_block_test.py's robust injector to perform replacement.
4) We write .generated/transformer_autogen.py and you can open it to verify.
"""
import os, sys, importlib.util, pathlib, json

ROOT = os.path.dirname(os.path.dirname(__file__))
tpl_candidates = [
    os.path.join(ROOT, "services", "templates", "transformer.slot.v1.j2"),
    os.path.join(ROOT, "transformer.slot.v1.j2"),
]
tpl = next((p for p in tpl_candidates if os.path.exists(p)), None)
if not tpl:
    print("[ERROR] transformer.slot.v1.j2 not found. Did you copy Step 1 templates?")
    sys.exit(1)

# 1) Minimal structured spec for transformer
spec = {
    "dims": {
        "num_layers": 6,  # legacy fallback
        "encoder_layers": {
            "count": 4,            # Try: 4 layers
            "name_fmt": "enc_{i}", # layer name format
            # "params": {"d_model": 768, "num_heads": 12, "ffn_dim": 3072, "dropout_rate": 0.1},
        }
    },
    # "custom_blocks": {}  # Intentionally empty; we'll auto-fill
}

# 2) Auto-fill custom_blocks
ab_spec_path = os.path.join(ROOT, "services", "codegen_autoblocks.py")
spec_mod = importlib.util.spec_from_file_location("codegen_autoblocks", ab_spec_path)
ab = importlib.util.module_from_spec(spec_mod)
sys.modules["codegen_autoblocks"] = ab
spec_mod.loader.exec_module(ab)
spec = ab.autofill_custom_blocks(spec, family="transformer")

# 3) Inject using robust injector
inj_path = os.path.join(ROOT, "tools", "smoke_custom_block_test.py")
inj_spec = importlib.util.spec_from_file_location("smoke_custom_block_test", inj_path)
inj = importlib.util.module_from_spec(inj_spec)
sys.modules["smoke_custom_block_test"] = inj
inj_spec.loader.exec_module(inj)

with open(tpl, "r", encoding="utf-8") as f:
    tpl_txt = f.read()

out_txt = inj.inject_custom_blocks(tpl_txt, spec.get("custom_blocks", {}))

out_path = os.path.join(ROOT, ".generated", "transformer_autogen.py")
pathlib.Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(out_txt)

print("✅ Auto-gen demo completed")
print("→ Output:", out_path)
print("Tip) Open the file and locate the 'for i in range(4):' block in encoder_layers slot.")
