# -*- coding: utf-8 -*-
"""
TensorFlow(Keras) Base Code Generator (LLM Fallback Only, MVP)
- 목적: model_extractor_agent가 준 (model_name, components, description)을 기반으로
        템플릿 없이 GPT-4.1에게 '단일 파일 Keras 스켈레톤 코드'만 생성하도록 요청.
- 설계 포인트:
  1) 프롬프트에서 출력 형식을 '코드만' 요구 (설명 금지)
  2) 학습루프/데이터셋 로딩 금지 → 최소한의 forward 테스트까지만
  3) __main__ 블록에서 더미 입력으로 model(x) 호출하여 shape 출력
  4) 재시도 로직과 간단한 사후 정제 포함
"""

from __future__ import annotations
from typing import Optional, List
import os
import re
import textwrap
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# -----------------------------
# 0) LLM 클라이언트 (Azure OpenAI GPT-4o)
# -----------------------------
_llm = AzureChatOpenAI(
    # azure_deployment=os.getenv("AOAI_DEPLOY_GPT4O"),
    # openai_api_version="2024-02-01",
    azure_deployment=os.getenv("AOAI_DEPLOY_GPT41"),
    openai_api_version="2024-10-21",
    api_key=os.getenv("AOAI_API_KEY"),
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    temperature=0.1,
)


# -----------------------------
# 1) Fallback 전용 프롬프트 (Keras)
# -----------------------------
# - 출력은 반드시 '코드만'
# - keras.Model/Sequential 사용
# - __main__에서 더미 입력 forward + shape 출력
# - 학습 루프/데이터 로딩/파일 저장 금지
# - 과한 주석, 긴 설명 금지
_FALLBACK_PROMPT = ChatPromptTemplate.from_template(
    """
아래 모델 설명을 바탕으로 TensorFlow(Keras) 단일 파일 base code를 생성하세요.

필수 조건:
- keras.Model 또는 keras.Sequential 사용
- __main__ 블록 포함: 더미 입력으로 model(x) 한 번 호출하여 출력 텐서 shape를 print
- 학습/컴파일/데이터셋 로딩 금지 (fit/compile/dataset 사용 금지)
- 외부 파일 입출력 금지
- 주석은 최소화
- 출력은 '코드만' 주세요. (설명/마크다운 금지)

모델 설명:
- name: {model_name}
- components: {components}
- description: {description}
""".strip()
)

_chain = _FALLBACK_PROMPT | _llm | StrOutputParser()


# -----------------------------
# 2) 간단 전처리/후처리 유틸
# -----------------------------
def _truncate(x: str, max_len: int = 2000) -> str:
    """LLM 프롬프트 입력 길이 방어용 슬라이싱."""
    if not x:
        return ""
    return x[:max_len]


def _strip_markdown_fences(code: str) -> str:
    """
    모델이 실수로 ```python ... ``` 처럼 감싸면 제거.
    """
    c = code.strip()
    # ```python ... ```
    c = re.sub(r"^\s*```(?:python)?\s*", "", c)
    c = re.sub(r"\s*```\s*$", "", c)
    return c.strip()


def _ensure_main_block(code: str) -> str:
    """
    __main__ 블록이 없으면 간단히 추가해준다.
    - 사용자 실행성을 최우선으로 보장
    """
    if "__main__" in code:
        return code
    # 아주 미니멀한 fallback main
    tail = textwrap.dedent(
        """
    if __name__ == "__main__":
        import numpy as np
        # 입력 예시는 보편적인 (B, T, F) 또는 (B, H, W, C) 중 하나로 가정
        try:
            x = np.random.randn(4, 100, 4).astype("float32")
        except Exception:
            x = np.random.randn(2, 32, 32, 3).astype("float32")
        y = model(x)
        try:
            print(getattr(model, "name", "model"), y.shape)
        except Exception as e:
            print("forward ok", type(y))
    """
    ).strip("\n")
    # model 변수가 없을 가능성도 있지만, LLM 프롬프트가 model 변수명을 맞추도록 유도함
    return code.rstrip() + "\n\n" + tail + "\n"


# -----------------------------
# 3) 공개 API
# -----------------------------
def generate_base_code(
    model_name: Optional[str],
    components: Optional[List[str]],
    description: Optional[str],
) -> str:
    """
    템플릿 없이 LLM에게 Keras 스켈레톤을 생성시킨다.
    - model_name/components/description은 문자열 슬라이싱 하여 입력
    - 출력은 코드만 반환 (마크다운 fences 제거 + __main__ 보강)
    """
    mn = model_name or "UnknownModel"
    cmps = ", ".join(components or [])
    desc = description or ""

    # 프롬프트 길이 가드
    mn = _truncate(mn, 200)
    cmps = _truncate(cmps, 1000)
    desc = _truncate(desc, 1500)

    # 1차 시도
    code = _chain.invoke(
        {"model_name": mn, "components": cmps, "description": desc}
    ).strip()

    # 코드 정리
    code = _strip_markdown_fences(code)
    code = _ensure_main_block(code)

    # 간단 유효성 체크: import tensorflow / from tensorflow import keras 중 하나는 있어야 함
    if ("import tensorflow" not in code) and ("from tensorflow" not in code):
        # 재시도 1회: 보다 강하게 지시
        strong_prompt = ChatPromptTemplate.from_template(
            _FALLBACK_PROMPT.template
            + "\n\n중요: 반드시 'import tensorflow as tf' 와 'from tensorflow import keras'를 포함할 것. 출력은 코드만."
        )
        strong_chain = strong_prompt | _llm | StrOutputParser()
        retry = strong_chain.invoke(
            {"model_name": mn, "components": cmps, "description": desc}
        ).strip()
        retry = _strip_markdown_fences(retry)
        retry = _ensure_main_block(retry)
        if ("import tensorflow" in retry) or ("from tensorflow" in retry):
            return retry
    return code
