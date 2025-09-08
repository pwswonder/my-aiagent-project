
"""
Smoke Test: CUSTOM_BLOCK slot injection (robust matcher)
--------------------------------------------------------
- Matches any of:
    # {{CUSTOM_BLOCK:slot}}
    # {CUSTOM_BLOCK:slot}
    # {% raw %}{{CUSTOM_BLOCK:slot}}{% endraw %}
  with or without spaces after '#', and with trailing comments.

Usage:
  python tools/smoke_custom_block_test.py \
    --template services/templates/transformer.j2 \
    --out .generated/transformer_smoke.py
"""
import argparse, os, re, sys, pathlib

# Robust single-regex that tolerates:
# - one or two braces
# - optional {% raw %} / {% endraw %}
# - optional trailing comments
# - no/any spaces after '#'
SLOT_LINE = re.compile(
    r"^(?P<indent>\s*)\#\s*"
    r"(?:\{\%\s*raw\s*\%\}\s*)?"          # optional {% raw %}
    r"\{\{?\s*CUSTOM_BLOCK:(?P<slot>[^}\s]+)\s*\}\}?"
    r"(?:\s*\{\%\s*endraw\s*\%\})?"       # optional {% endraw %}
    r"(?:\s*(?P<trail>\#.*))?$",          # optional trailing comment
    re.MULTILINE
)

def inject_custom_blocks(template_text: str, custom_blocks: dict) -> str:
    def _replace(m: re.Match) -> str:
        indent = m.group("indent") or ""
        slot = (m.group("slot") or "").strip()
        code = (custom_blocks or {}).get(slot, "")
        if not code:
            # Keep original marker line if no code provided
            return m.group(0)
        # Preserve indentation for multi-line code
        lines = [indent + ln for ln in code.splitlines()]
        return '\n'.join(lines)

    return SLOT_LINE.sub(_replace, template_text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="*.j2 template path")
    ap.add_argument("--out", default=".generated/slot_smoke_output.py", help="output .py file")
    ap.add_argument("--only", default="", help="comma-separated slot names to inject (optional)")
    args = ap.parse_args()

    tpl_path = args.template
    out_path = args.out
    only = [s.strip() for s in args.only.split(",") if s.strip()]

    if not os.path.exists(tpl_path):
        print(f"[ERROR] Template not found: {tpl_path}")
        sys.exit(1)

    # Demo custom_blocks. Replace with real layer code as needed.
    default_blocks = {
        "encoder_layers": "print('ENCODER_LAYERS_INJECTED - demo')",
        "decoder_layers": "print('DECODER_LAYERS_INJECTED - demo')",
        "head": "print('HEAD_INJECTED - demo')",
        "stages": "print('STAGES_INJECTED - demo')",
        "rnn_stack": "print('RNN_STACK_INJECTED - demo')",
        "encoder": "print('AE/VAE ENCODER_INJECTED - demo')",
        "latent": "print('VAE LATENT_INJECTED - demo')",
        "decoder": "print('AE/VAE DECODER_INJECTED - demo')",
        "encoder_blocks": "print('UNET ENCODER_BLOCKS_INJECTED - demo')",
        "decoder_blocks": "print('UNET DECODER_BLOCKS_INJECTED - demo')",
        "generator_blocks": "print('GAN GENERATOR_BLOCKS_INJECTED - demo')",
        "discriminator_block": "print('GAN DISCRIMINATOR_BLOCK_INJECTED - demo')",
        "gan_loss": "print('GAN LOSS_INJECTED - demo')",
    }
    if only:
        custom_blocks = {k: v for k, v in default_blocks.items() if k in only}
    else:
        custom_blocks = default_blocks

    with open(tpl_path, "r", encoding="utf-8") as f:
        tpl = f.read()

    out_text = inject_custom_blocks(tpl, custom_blocks)

    pathlib.Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_text)

    # Basic report
    matched_slots = set(m.group("slot") for m in SLOT_LINE.finditer(tpl))
    injected_slots = [s for s in matched_slots if s in custom_blocks]
    print("✅ Smoke test completed")
    print(f"→ Input : {tpl_path}")
    print(f"→ Output: {out_path}")
    print(f"Matched slots   : {sorted(matched_slots)}")
    print(f"Injected slots  : {sorted(injected_slots)}")
    print("Tip) 결과 파일에서 'INJECTED - demo' 를 검색해 주세요.")

if __name__ == "__main__":
    main()
