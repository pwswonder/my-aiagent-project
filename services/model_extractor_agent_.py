"""
services/model_extractor_agent.py

Phase 1: 논문 텍스트에서 '제안 모델' 스펙(ModelSpec)을 구조화하여 추출.
- LangChain + AzureChatOpenAI 기반 Structured Output(JSON)
- 추출 후 spec_verifier로 검증/보정
- 이후 단계(템플릿 선택/코드 생성)로 넘길 '단일 진실 소스' 제공

"""

from __future__ import annotations
import os
import json
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .spec_schema import ModelSpec
from .spec_verifier import verify_and_normalize

# ------------------------------------------------------------
# 0) 환경설정
# ------------------------------------------------------------
load_dotenv()

_llm = AzureChatOpenAI(
    # azure_deployment=os.getenv("AOAI_DEPLOY_GPT4O"),
    # openai_api_version="2024-02-01",
    azure_deployment=os.getenv("AOAI_DEPLOY_GPT41"),
    openai_api_version="2024-10-21",
    api_key=os.getenv("AOAI_API_KEY"),
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    temperature=0.0
)

# ------------------------------------------------------------
# 1) 프롬프트(제안 모델만! baseline과 혼동 금지)
# ------------------------------------------------------------
_SYSTEM = """You are an expert research paper information extractor.
ONLY extract the SPEC of the PROPOSED model in the paper (NOT baselines).
Return a STRICT JSON according to the schema.
If uncertain, set confidence low and fill `is_proposed_clearly_identified=false`.
Include short evidence snippets that justify the proposed model family/subtype.
"""

# 가능한 값 힌트(LLM에 제공해 오류율↓)
_HINTS = """
Valid task_type examples: ["time_series_forecasting","classification","regression","machine_translation","text_summarization","qa","image_classification","object_detection","segmentation","speech_recognition","recommendation","other"].
Valid data_modality examples: ["time_series","text","image","audio","tabular","graph","multimodal","other"].
Valid proposed_model_family: ["Transformer","Linear","DLinear","NLinear","CNN","RNN","LSTM","GRU","TCN","GNN","MLP","ARIMA","Prophet","S4","Hybrid","Other"].
Valid transformer subtypes: ["Encoder","Decoder","EncoderDecoder","Informer","Autoformer","Reformer","Linformer","Longformer","Perceiver","ViT","PatchTST","TST","Other"].
Valid linear subtypes: ["DLinear","NLinear","PatchTST-Linear","Other"].
Valid objective: ["mse","mae","cross_entropy","binary_cross_entropy","huber","logcosh","other"].
Valid positional_encoding: ["absolute","relative","none","other"].
"""

# 스키마 예시(JSON 포맷 템플릿) – 파싱 안정화용
_JSON_SCHEMA_EXAMPLE = r"""
{
  "title": "<string or null>",
  "task_type": "<TaskType>",
  "data_modality": "<DataModality>",
  "proposed_model_family": "<ModelFamily>",
  "subtype": "<string or null>",
  "key_blocks": ["<string>", "..."],
  "dims": {
    "in_dim": <int or null>,
    "out_dim": <int or null>,
    "seq_len": <int or null>,
    "pred_len": <int or null>,
    "hidden_dim": <int or null>,
    "num_layers": <int or null>,
    "num_heads": <int or null>,
    "ffn_dim": <int or null>,
    "kernel_size": <int or null>,
    "dilation": <int or null>,
    "dropout": <float or null>
  },
  "objective": "<ObjectiveType or null>",
  "positional_encoding": "<PositionalEncodingType or null>",
  "baselines": [
    {"name": "<string>", "family": "<ModelFamily or null>", "notes": "<string or null>"}
  ],
  "evidence": [
    {"text": "<short quote>", "section": "<string or null>", "page": <int or null>}
  ],
  "confidence": <float between 0 and 1>,
  "is_proposed_clearly_identified": <true|false>
}
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("user", """Extract the proposed model SPEC from the following paper content.

# HINTS
{hints}

# REQUIRED JSON SCHEMA (EXAMPLE FORMAT)
{json_schema}

# PAPER CONTENT
Title (if known): {title}
---
{paper_text}
"""),
])

_parser = StrOutputParser()

# ------------------------------------------------------------
# 2) 유틸: JSON 안전 파싱
# ------------------------------------------------------------
def _safe_json_loads(s: str) -> Dict[str, Any]:
    """
    모델이 반환한 문자열에서 첫번째 JSON 오브젝트 블록만 파싱.
    - fence/설명문 섞여 나오는 경우 대비
    """
    s = s.strip()
    # 가장 간단한 전략: 첫 '{' ~ 마지막 '}' 범위 추출
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]
    return json.loads(s)

# ------------------------------------------------------------
# 3) 외부 인터페이스
# ------------------------------------------------------------
def extract_model_spec(paper_text: str, title: Optional[str] = None) -> Dict[str, Any]:
    """
    입력: 논문 본문(또는 요약) 텍스트
    출력: { "raw": <ModelSpec dict>, "verified": <VerifiedSpec dict> }
    - raw: LLM이 생성한 스펙(JSON) → ModelSpec로 강제 파싱
    - verified: 검증/보정 결과(경고 포함)
    """
    # 1) LLM 호출
    chain = _PROMPT | _llm | _parser
    out = chain.invoke({
        "hints": _HINTS,
        "json_schema": _JSON_SCHEMA_EXAMPLE,
        "paper_text": paper_text[:40000],  # 과도한 길이 방지(필요 시 슬라이딩 윈도)
        "title": title or "(unknown)",
    })

    # 2) JSON 파싱 → ModelSpec
    raw_json = _safe_json_loads(out)
    raw_spec = ModelSpec.model_validate(raw_json)

    # 3) 검증/보정
    verified = verify_and_normalize(raw_spec)

    # 4) 사전형으로 반환(후단 서비스들에서 쉽게 사용)
    return {
        "raw": json.loads(raw_spec.model_dump_json()),
        "verified": json.loads(verified.model_dump_json())
    }