
# tests/test_routing_smoke.py
# 목적: routing.resolve_template_from_spec 스모크 테스트
# 실행: pytest -q tests/test_routing_smoke.py
import os, sys, json
from typing import Dict, Any

# 프로젝트 루트(예: services/*.py 가 위치한 경로)를 sys.path 에 추가
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import routing  # /mnt/data/routing.py

def _mk_spec(
    family: str = "",
    subtype: str = "",
    task: str = "",
    modality: str = "",
    title: str = "",
    evidence_texts=None,
    notes: str = "",
    baselines=None,
) -> Dict[str, Any]:
    """테스트용 spec 최소 구성 생성 유틸리티"""
    return {
        "proposed_model_family": family,
        "subtype": subtype,
        "task_type": task,
        "data_modality": modality,
        "title": title,
        "notes": notes,
        # routing.py 에서 evidence 는 list[dict(text=...)] 또는 list[str] 모두 허용
        "evidence": [{"text": t} for t in (evidence_texts or [])],
        "baselines": baselines or [],
    }

def test_transformer_mt():
    spec = _mk_spec(
        family="Transformer",
        subtype="EncoderDecoder",
        task="Machine_Translation",
        modality="Text",
        title="Neural Machine Translation with Self-Attention",
        evidence_texts=["self-attention", "multi-head attention", "seq2seq"],
    )
    key, meta = routing.resolve_template_from_spec(spec)
    assert key == "transformer", f"expected transformer, got {key} (meta={meta})"
    assert meta["score"] > 0 and meta["rule_index"] is not None

def test_resnet_classification():
    spec = _mk_spec(
        family="ResNet",
        task="Image_Classification",
        modality="Image",
        title="Residual Networks for Image Classification",
        evidence_texts=["residual", "skip connection"],
    )
    key, meta = routing.resolve_template_from_spec(spec)
    assert key == "resnet", f"expected resnet, got {key} (meta={meta})"
    assert meta["score"] > 0 and meta["rule_index"] is not None

def test_unet_segmentation():
    spec = _mk_spec(
        family="U-Net",
        task="Segmentation",
        modality="Image",
        title="U-Net: Convolutional Networks for Biomedical Image Segmentation",
        evidence_texts=["u-net", "segmentation", "pixel-wise"],
    )
    key, meta = routing.resolve_template_from_spec(spec)
    assert key == "unet", f"expected unet, got {key} (meta={meta})"
    assert meta["score"] > 0 and meta["rule_index"] is not None

def test_rnn_lstm_sequence():
    spec = _mk_spec(
        family="RNN",
        subtype="LSTM",
        task="Sequence_Modeling",
        modality="Text",
        title="Recurrent neural network with LSTM",
        evidence_texts=["recurrent", "lstm"],
    )
    key, meta = routing.resolve_template_from_spec(spec)
    # 라우팅 규칙은 'rnn_seq' 템플릿을 반환하도록 정의되어 있음
    assert key == "rnn_seq", f"expected rnn_seq, got {key} (meta={meta})"
    assert meta["score"] > 0 and meta["rule_index"] is not None

def test_ambiguous_fallback():
    # 의도적으로 모호: family/task/keywords 거의 없음
    spec = _mk_spec(
        family="UnknownFamily",
        task="other",
        modality="tabular",
        title="A study on something unclear",
        evidence_texts=["novel approach with mixed signals"],
    )
    key, meta = routing.resolve_template_from_spec(spec)
    # routing.py 의 기본 폴백은 'mlp'
    assert key == "mlp", f"expected mlp fallback, got {key} (meta={meta})"
    assert meta["rule_index"] is None  # 매칭 규칙 없음
