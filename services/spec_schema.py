"""
services/spec_schema.py

Phase 1: 모델 스펙(구조/하이퍼파라미터/증거)을 엄격한 Pydantic v2 스키마로 정의.
- LLM 출력(JSON)을 이 스키마로 파싱하여 타입 안정성 확보
- 이후 템플릿 선택, 코드 합성, 셀프체크의 신뢰도를 높이기 위함
"""

from __future__ import annotations
from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field, field_validator

# ------------------------------------------------------------
# 0) 공통 타입(택소노미) 정의
# ------------------------------------------------------------
TaskType = Literal[
    # 예측/생성 태스크 (필요시 추가)
    "time_series_forecasting", "classification", "regression",
    "machine_translation", "text_summarization", "qa",
    "image_classification", "object_detection", "segmentation",
    "speech_recognition", "recommendation", "other",
]

DataModality = Literal[
    "time_series", "text", "image", "audio", "tabular", "graph", "multimodal", "other"
]

ModelFamily = Literal[
    "Transformer", "Linear", "DLinear", "NLinear", "CNN", "RNN", "LSTM", "GRU",
    "TCN", "GNN", "MLP", "ARIMA", "Prophet", "S4", "Hybrid", "Other"
]

TransformerSubtype = Literal[
    "Encoder", "Decoder", "EncoderDecoder", "Informer", "Autoformer", "Reformer",
    "Linformer", "Longformer", "Perceiver", "ViT", "PatchTST", "TST", "Other"
]

LinearSubtype = Literal[
    "DLinear", "NLinear", "PatchTST-Linear", "Other"
]

ObjectiveType = Literal[
    "mse", "mae", "cross_entropy", "binary_cross_entropy", "huber", "logcosh", "other"
]

PositionalEncodingType = Literal[
    "absolute", "relative", "none", "other"
]

# ------------------------------------------------------------
# 1) 보조 스키마
# ------------------------------------------------------------
class BaselineModel(BaseModel):
    """논문에서 비교에 사용된 기준모델 정보(필요시 템플릿 가중치에 활용)"""
    name: str = Field(..., description="베이스라인 모델명(예: 'ARIMA', 'LSTM', 'Transformer')")
    family: Optional[ModelFamily] = Field(None, description="모델 계열 추정")
    notes: Optional[str] = Field(None, description="세부 설명/하이퍼파라미터 등")

class EvidenceSnippet(BaseModel):
    """제안 모델을 특정하는 증거(원문 스팬)"""
    text: str = Field(..., description="원문 인용(짧게)")
    section: Optional[str] = Field(None, description="절 제목(있다면)")
    page: Optional[int] = Field(None, description="페이지 번호(있다면)")

class DimensionConfig(BaseModel):
    """주요 차원/하이퍼파라미터(존재하는 값만 채움)"""
    in_dim: Optional[int] = None
    out_dim: Optional[int] = None
    seq_len: Optional[int] = None
    pred_len: Optional[int] = None
    hidden_dim: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    ffn_dim: Optional[int] = None
    kernel_size: Optional[int] = None
    dilation: Optional[int] = None
    dropout: Optional[float] = None

# ------------------------------------------------------------
# 2) 메인 스키마: ModelSpec
# ------------------------------------------------------------
class ModelSpec(BaseModel):
    """
    논문에서 '제안된 모델'의 구조를 기술하는 표준 스키마.
    - 모든 생성 단계의 입력(= 단일 진실 소스)
    """
    # (A) 메타
    title: Optional[str] = Field(None, description="논문 제목")
    task_type: TaskType = "other"
    data_modality: DataModality = "other"

    # (B) 제안 모델 계열/하위유형
    proposed_model_family: ModelFamily = "Other"
    subtype: Optional[str] = Field(None, description="세부 유형(예: EncoderDecoder, DLinear, PatchTST 등)")

    # (C) 핵심 구성요소/블록(코드 셀프체크에 사용)
    key_blocks: List[str] = Field(default_factory=list, description="예: ['MultiHeadAttention','LayerNorm','Residual']")

    # (D) 차원/하이퍼파라미터
    dims: DimensionConfig = Field(default_factory=DimensionConfig)

    # (E) 목적함수/포지셔널 인코딩 등
    objective: Optional[ObjectiveType] = None
    positional_encoding: Optional[PositionalEncodingType] = None

    # (F) 비교모델/근거
    baselines: List[BaselineModel] = Field(default_factory=list)
    evidence: List[EvidenceSnippet] = Field(default_factory=list)

    # (G) 신뢰도
    confidence: float = 0.0

    # (H) 내부 플래그
    is_proposed_clearly_identified: bool = False

    # ---------------- Validators ----------------
    @field_validator("confidence")
    @classmethod
    def _clip_confidence(cls, v: float) -> float:
        # 0~1 사이로 클램프
        if v < 0.0: return 0.0
        if v > 1.0: return 1.0
        return v

    @field_validator("key_blocks")
    @classmethod
    def _strip_blocks(cls, v: List[str]) -> List[str]:
        # 공백/중복 정리
        out: List[str] = []
        seen = set()
        for s in v:
            s2 = (s or "").strip()
            if s2 and s2.lower() not in seen:
                seen.add(s2.lower())
                out.append(s2)
        return out

# ------------------------------------------------------------
# 3) 검증 이후 통신 포맷(경고/수정 결과)
# ------------------------------------------------------------
class VerificationWarning(BaseModel):
    code: str
    message: str
    fix_applied: bool = False

class VerifiedSpec(BaseModel):
    spec: ModelSpec
    warnings: List[VerificationWarning] = Field(default_factory=list)
