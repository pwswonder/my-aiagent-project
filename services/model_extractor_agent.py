from __future__ import annotations
from typing import Dict, Any
import os
import re
import json

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ------------------------------------------------------------
# 0) 환경 변수 로드
# ------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------
# 1) LLM 클라이언트 (Azure OpenAI GPT-4o)
#    - temperature=0.0: 정보 추출 일관성↑
# ------------------------------------------------------------
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
# 2) 섹션 추출 로직
#    - 1순위: Proposed Method / Model / Architecture / Methodology / Approach 섹션 통짜 추출
#    - 2순위: 모델 연관 키워드 포함 문단들 상위 N개
#    - 3순위: raw_text 앞부분 일부
# ------------------------------------------------------------

# (1) '섹션 헤더'를 탐지하기 위한 정규표현식 (숫자 접두/대문자 헤더 모두)
_SECTION_PAT = re.compile(
    r"""
    (?:^|\n|\r)                                 # 줄 시작
    \s*(?:\d+\s*[\.\)]\s*)?                     # '3.' 또는 '3)' 같은 번호 헤더 (선택)
    (?:                                         # 섹션 제목 후보
        proposed\s+method|
        model(?:\s+architecture)?|
        architecture|
        methodology|
        approach|
        method
    )
    \b                                          # 단어 경계
    .*?                                         # 해당 헤더부터의 본문
    (?=                                         # 다음 섹션의 시작을 만날 때까지 (전방탐색)
        \n\s*\d+\s*[\.\)]\s*[A-Z]               # '4. X' 같은 다음 번호 헤더
        | \n\s*[A-Z][^\n]{0,60}\n              # 또는 줄 단위의 대문자 헤더
        | \Z                                   # 또는 텍스트 끝
    )
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

# (2) '모델 관련 키워드'가 포함된 문단 필터
#     - 범용 키워드: model, architecture, encoder/decoder, transformer, attention, cnn, rnn, lstm, gan, bert, vit, gpt 등
_KW_PAT = re.compile(
    r"(?:\bmodel\b|\barchitecture\b|encoder|decoder|autoencoder|\bae\b|transformer|attention|\bcnn\b|\brnn\b|\blstm\b|\bgan\b|\bbert\b|\bvit\b|\bgpt\b|\bresnet\b|\bbackbone\b|\bmlp\b)",
    re.IGNORECASE,
)

def _extract_model_section(raw_text: str, max_paragraphs: int = 10, fallback_chars: int = 2000) -> str:
    """
    주 텍스트에서 '모델 설명' 섹션을 최대한 정확히 잘라낸다.
    1) 섹션 헤더 패턴에 매칭되는 첫 블록
    2) 키워드 포함 문단 상위 N개
    3) 최후 fallback: 앞부분 일부
    """
    if not raw_text:
        return ""

    # 1) 섹션 헤더 매칭
    m = _SECTION_PAT.search(raw_text)
    if m:
        section = m.group(0).strip()
        # 너무 긴 섹션이면 과도한 토큰 방지를 위해 앞쪽만 사용
        if len(section) > 8000:
            return section[:8000]
        return section

    # 2) 키워드 기반 문단 추출 (상위 N개)
    paragraphs = [p.strip() for p in raw_text.split("\n") if p.strip()]
    picked = [p for p in paragraphs if _KW_PAT.search(p)]
    if picked:
        joined = "\n".join(picked[:max_paragraphs])
        return joined[:8000]  # 안전 가드

    # 3) 앞부분 일부
    return raw_text[:fallback_chars]


# ------------------------------------------------------------
# 3) 프롬프트 정의 (JSON 강제)
#    - LLM이 JSON 이외의 것을 내지 않도록 명확히 요구
# ------------------------------------------------------------
_PROMPT = ChatPromptTemplate.from_template(
    """
당신은 AI 논문 분석 전문가입니다.
아래 텍스트는 논문에서 모델을 설명하는 부분(또는 관련 문단)입니다.
이 텍스트를 바탕으로 '모델 정보'를 JSON으로 정리하세요.

요구사항:
- model_name: 모델 이름 또는 합리적인 추정(없으면 null)
- components: 핵심 구성 요소 문자열 배열 (예: ["CNN backbone","Transformer encoder","Attention","MLP head"])
- description: 한 문장 요약 (모델의 핵심 아이디어/동작)

텍스트:
{model_section}

반드시 아래 JSON 스키마로만 출력:
{{
  "model_name": string or null,
  "components": string[],
  "description": string
}}
""".strip()
)

# LLM 체인 (프롬프트 → GPT-4o → 문자열 출력)
_chain = _PROMPT | _llm | StrOutputParser()

# ------------------------------------------------------------
# 4) 외부에서 호출하는 함수
#    - LangGraph 노드에서 이 함수만 호출하면 됨
# ------------------------------------------------------------
def run_model_extractor(raw_text: str) -> Dict[str, Any]:
    """
    raw_text에서 모델 설명 섹션을 추출하고, LLM으로 모델 정보를 JSON 형태로 파싱한다.

    Args:
        raw_text (str): 논문 전체 원문 텍스트 (추출/전처리된 문자열)

    Returns:
        Dict[str, Any]: {
            "model_name": str | None,
            "components": List[str],
            "description": str
        }
    """
    try:
        section = _extract_model_section(raw_text or "")
        # LLM 호출 (길이 초과 방지: 섹션은 이미 8k로 방어됨)
        out_str = _chain.invoke({"model_section": section}).strip()
        try:
            data = json.loads(out_str)
        except json.JSONDecodeError:
            # 혹시 모델이 포맷을 어겼다면, 최소한의 형태로 감싸서 반환
            data = {
                "model_name": None,
                "components": [],
                "description": out_str,
            }

        # 필드 보정 (누락 시 기본값)
        if "model_name" not in data:
            data["model_name"] = None
        if "components" not in data or not isinstance(data["components"], list):
            data["components"] = []
        if "description" not in data or not isinstance(data["description"], str):
            data["description"] = ""

        return data

    except Exception as e:
        # 에러 시에도 호출 측이 안전하게 처리할 수 있도록 기본 형태 반환
        return {
            "model_name": None,
            "components": [],
            "description": f"[model_extractor_error] {e}"
        }


# ------------------------------------------------------------
# 5) 로컬 단위 테스트 (직접 실행 시)
#    - python services/model_extractor_agent.py 로 확인 가능
# ------------------------------------------------------------
if __name__ == "__main__":
    demo_text = """
    3. Proposed Method
    We propose a hybrid architecture combining a CNN backbone with a Transformer encoder.
    The CNN extracts local spatial features, while the self-attention layers model global dependencies.
    Finally, an MLP head produces the logits. Our approach improves robustness on image benchmarks.
    """
    result = run_model_extractor(demo_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
