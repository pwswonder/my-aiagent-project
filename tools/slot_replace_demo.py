
"""
Indent-preserving slot replacement demo.

- Reads a slotted template (already rendered Jinja or plain .j2).
- Replaces lines like:   [indent]# {% raw %}{{CUSTOM_BLOCK:slot}}{% endraw %}
  with user-provided code, re-indenting each replacement line to the same indent.
- Shows a short before/after diff.
"""

from pathlib import Path
import re
import sys
import difflib

SLOT_RE = re.compile(r"^([ \t]*)#\s*\{\%\s*raw\s*\%\}\{\{CUSTOM_BLOCK:([a-zA-Z0-9_]+)\}\}\{\%\s*endraw\s*\%\}\s*$", re.M)

def _indent_block(code: str, indent: str) -> str:
    """Add indent string to every non-empty line in code. Preserve trailing newline."""
    lines = code.splitlines(True)
    out = []
    for ln in lines:
        if ln.strip():
            out.append(indent + ln)
        else:
            out.append(ln)
    return "".join(out)

def replace_slots(text: str, mapping: dict) -> str:
    def _repl(m):
        indent, slot = m.group(1), m.group(2)
        payload = mapping.get(slot, "")  # empty means delete the slot line
        if not payload:
            return ""  # remove the line entirely
        return _indent_block(payload.rstrip() + "\n", indent)
    return SLOT_RE.sub(_repl, text)

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/slot_replace_demo.py /path/to/template.j2")
        sys.exit(1)

    p = Path(sys.argv[1])
    t = p.read_text(encoding="utf-8")

    # Example mapping for demo purposes only
    mapping = {
        "imports_extra": "import math\n",
        "compile_override": "# override or add compile options here\n# model.compile(optimizer='adam', loss='mse', metrics=['mae'])",
        "callbacks": "callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]",
        "se_squeeze_excitation": "x = SEBlock(channels=x.shape[-1])(x)",
        "transformer_encoder_custom": "# add rotary embeddings / pre-norm, etc.",
        "rnn_custom": "x = layers.Dropout(0.2)(x)",
    }

    out = replace_slots(t, mapping)
    diff = difflib.unified_diff(t.splitlines(True), out.splitlines(True), fromfile="before", tofile="after")
    sys.stdout.writelines(diff)

if __name__ == "__main__":
    main()
