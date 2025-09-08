# tools/run_routing_smoke.py
# 목적: pytest 없이도 스모크 테스트를 단독 실행할 수 있도록 하는 러너
import os, sys, traceback
from typing import Dict, Any

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from services.routing import resolve_template_from_spec


def _mk_spec(
    family="",
    subtype="",
    task="",
    modality="",
    title="",
    evidence_texts=None,
    notes="",
    baselines=None,
):
    return {
        "proposed_model_family": family,
        "subtype": subtype,
        "task_type": task,
        "data_modality": modality,
        "title": title,
        "notes": notes,
        "evidence": [{"text": t} for t in (evidence_texts or [])],
        "baselines": baselines or [],
    }


CASES = [
    (
        "Transformer(MT)",
        _mk_spec(
            family="Transformer",
            subtype="EncoderDecoder",
            task="Machine_Translation",
            modality="Text",
            title="Neural Machine Translation with Self-Attention",
            evidence_texts=["self-attention", "multi-head attention", "seq2seq"],
        ),
        "transformer",
    ),
    (
        "ResNet(cls)",
        _mk_spec(
            family="ResNet",
            task="Image_Classification",
            modality="Image",
            title="Residual Networks for Image Classification",
            evidence_texts=["residual", "skip connection"],
        ),
        "resnet",
    ),
    (
        "U-Net(seg)",
        _mk_spec(
            family="U-Net",
            task="Segmentation",
            modality="Image",
            title="U-Net: Convolutional Networks for Biomedical Image Segmentation",
            evidence_texts=["u-net", "segmentation", "pixel-wise"],
        ),
        "unet",
    ),
    (
        "RNN(LSTM)",
        _mk_spec(
            family="RNN",
            subtype="LSTM",
            task="Sequence_Modeling",
            modality="Text",
            title="Recurrent neural network with LSTM",
            evidence_texts=["recurrent", "lstm"],
        ),
        "rnn_seq",
    ),
    (
        "Ambiguous(fallback)",
        _mk_spec(
            family="UnknownFamily",
            task="other",
            modality="tabular",
            title="A study on something unclear",
            evidence_texts=["novel approach with mixed signals"],
        ),
        "mlp",
    ),
]


def main():
    ok = True
    for name, spec, expected in CASES:
        try:
            # key, meta = routing.resolve_template_from_spec(spec)
            key, meta = resolve_template_from_spec(spec)

            mark = "✅" if key == expected else "❌"
            print(
                f"{mark} {name:>20}: got={key:12s} expected={expected:12s}  score={meta.get('score')} rule_index={meta.get('rule_index')}"
            )
            if key != expected:
                ok = False
        except Exception:
            ok = False
            print(f"❌ {name:>20}: EXCEPTION\n{traceback.format_exc()}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
