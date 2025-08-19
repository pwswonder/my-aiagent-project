"""
services/spec_verifier.py

범용 택소노미 검증기:
- 공통 검증(제안모델 식별, 모달리티/태스크 일관성, 증거 힌트)
- 패밀리별 요구 블록/권고 블록 확인(레지스트리 패턴)
- 간단한 휴리스틱 보정(과잉 보정 금지)

필요 시 templates/registry.yaml 등 외부 설정으로 required/optional 블록을 주입하도록 확장 가능.
"""

from __future__ import annotations
from typing import List, Dict, Callable, Set
from .spec_schema import ModelSpec, VerifiedSpec, VerificationWarning

# ------------------------------------------------------------
# 0) 패밀리별 요구 블록(기본값)
#    - 실제 프로젝트에서는 YAML/JSON로 분리하여 로드하는 것을 권장
# ------------------------------------------------------------
REQUIRED_BLOCKS_BY_FAMILY: Dict[str, Set[str]] = {
    "Transformer": {"multiheadattention", "layernorm"},
    "CNN": {"conv2d"},         # image 기본 예시
    "TCN": {"conv1d"},         # dilation은 optional 체크로 분리 가능
    "RNN": {"rnn"},
    "LSTM": {"lstm"},
    "GRU": {"gru"},
    "GNN": {"graphconv", "gat", "gcn"},   # 구현에 따라 다름 → OR 매칭
    "MLP": {"dense"},
    "DLinear": {"linear"},     # 시계열 linear류
    "NLinear": {"linear"},
    "S4": {"s4layer"},         # 예시 레이블
}

OPTIONAL_BLOCKS_BY_FAMILY: Dict[str, Set[str]] = {
    "Transformer": {"residual", "feedforward", "positionalencoding"},
    "TCN": {"dilation", "causal"},
    "CNN": {"batchnorm", "maxpool"},
}

# ------------------------------------------------------------
# 1) 레지스트리: 패밀리별 세부 검증 함수(선택)
# ------------------------------------------------------------
FamilyValidator = Callable[[ModelSpec, List[VerificationWarning]], None]
FAMILY_VALIDATORS: Dict[str, FamilyValidator] = {}

def register_family_validator(family: str):
    def _wrap(fn: FamilyValidator):
        FAMILY_VALIDATORS[family.lower()] = fn
        return fn
    return _wrap

# 예: Transformer 전용 세부 검증(너무 구체화하지 않고 보편 요소만)
@register_family_validator("Transformer")
def _validate_transformer(spec: ModelSpec, warns: List[VerificationWarning]) -> None:
    blocks = {b.lower() for b in spec.key_blocks}
    if not any("multiheadattention" in b for b in blocks):
        warns.append(VerificationWarning(
            code="TRANSFORMER_MHA_MISSING",
            message="Transformer인데 MultiHeadAttention 관련 블록이 감지되지 않았습니다.",
            fix_applied=False,
        ))

@register_family_validator("TCN")
def _validate_tcn(spec: ModelSpec, warns: List[VerificationWarning]) -> None:
    # dilation 존재시 더 ‘TCN스러움’ → 없다고 경고까지는 X (참고정보용)
    blocks = {b.lower() for b in spec.key_blocks}
    if "conv1d" in blocks and "dilation" not in blocks:
        warns.append(VerificationWarning(
            code="TCN_WITHOUT_DILATION_HINT",
            message="TCN으로 보이지만 dilation 관련 단서가 없습니다(정보성 경고).",
            fix_applied=False,
        ))

# ------------------------------------------------------------
# 2) 공통 검증/보정
# ------------------------------------------------------------
_TS_HINTS = {"time series", "forecast", "long-term forecasting", "seq2pred", "temporal"}
_LINEAR_HINTS = {"dlinear", "nlinear", "linear", "moving average", "series decomposition"}

def _common_checks(spec: ModelSpec) -> List[VerificationWarning]:
    warns: List[VerificationWarning] = []

    # (A) 제안 모델 식별
    if not spec.is_proposed_clearly_identified:
        warns.append(VerificationWarning(
            code="PROPOSED_UNCLEAR",
            message="제안 모델(proposed) 식별이 불명확합니다. 증거/본문 재확인 권장.",
            fix_applied=False
        ))

    # (B) 모달리티-태스크 최소 일관성(강제 아님)
    if spec.data_modality == "time_series" and spec.task_type not in ("time_series_forecasting", "regression", "other"):
        warns.append(VerificationWarning(
            code="MODALITY_TASK_MISMATCH",
            message=f"time_series 모달리티인데 task_type={spec.task_type}. 재확인 권장.",
            fix_applied=False
        ))

    # (C) 패밀리별 필수 블록 존재성(너무 강제하지 않음; 경고 위주)
    fam = spec.proposed_model_family
    blocks = {b.lower() for b in spec.key_blocks}
    req = REQUIRED_BLOCKS_BY_FAMILY.get(fam, set())
    if req and not any(any(r in b for r in req) for b in blocks):
        warns.append(VerificationWarning(
            code="REQUIRED_BLOCKS_MISSING",
            message=f"{fam} 계열 핵심 블록 단서가 희박합니다(요구: {sorted(req)}).",
            fix_applied=False
        ))

    return warns

def _soft_coercions(spec: ModelSpec, warns: List[VerificationWarning]) -> None:
    """
    보수적 보정: ‘명백히’ 어긋난 경우만 살짝 보정.
    과잉 보정을 피하기 위해 조건을 엄격하게 둔다.
    """
    evidence_text = " ".join([e.text.lower() for e in spec.evidence])

    # 시계열 + linear 단서가 매우 강하고, family가 Other/MLP/RNN 계열이면 DLinear로 권고 보정
    if spec.data_modality == "time_series" and (
        any(k in evidence_text for k in _LINEAR_HINTS) or any(k in evidence_text for k in _TS_HINTS)
    ):
        if spec.proposed_model_family in ("Other", "MLP", "RNN", "LSTM", "GRU"):
            spec.proposed_model_family = "DLinear"
            if not spec.subtype:
                spec.subtype = "DLinear"
            warns.append(VerificationWarning(
                code="COERCE_TO_DLINEAR",
                message="시계열+Linear 단서가 강하고 family가 불명확하여 DLinear로 보정했습니다.",
                fix_applied=True
            ))

# ------------------------------------------------------------
# 3) 메인 엔트리포인트
# ------------------------------------------------------------
def verify_and_normalize(spec: ModelSpec) -> VerifiedSpec:
    warnings: List[VerificationWarning] = []

    # 공통 검증
    warnings.extend(_common_checks(spec))

    # 패밀리별 세부 검증 (등록된 경우)
    validator = FAMILY_VALIDATORS.get(spec.proposed_model_family.lower())
    if validator:
        validator(spec, warnings)

    # 소프트 보정(모달리티·증거 기반)
    _soft_coercions(spec, warnings)

    return VerifiedSpec(spec=spec, warnings=warnings)
