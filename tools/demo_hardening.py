
# tools/demo_hardening.py
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from services.spec_hardener import harden_spec_for_template

def show(name, spec, tk):
    s2, warns = harden_spec_for_template(spec, tk)
    print("="*80)
    print(f"[{name}] template={tk}")
    print("- before:")
    print(json.dumps(spec, ensure_ascii=False, indent=2))
    print("- after:")
    print(json.dumps(s2, ensure_ascii=False, indent=2))
    print("- warnings:")
    for w in warns:
        print("  -", w)

if __name__ == "__main__":
    show("Transformer MT", {"task_type":"Machine_Translation","proposed_model_family":"Transformer"}, "transformer")
    show("ResNet cls", {"task_type":"Image_Classification","proposed_model_family":"ResNet"}, "resnet")
    show("U-Net seg", {"task_type":"Segmentation","proposed_model_family":"U-Net"}, "unet")
    show("RNN LSTM", {"task_type":"Sequence_Modeling","proposed_model_family":"RNN","subtype":"LSTM"}, "rnn_seq")
    show("Ambiguous", {"task_type":"other"}, "mlp")


