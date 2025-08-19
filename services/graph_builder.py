# services/graph_builder.py
# ------------------------------------------------------------
# LangGraph 파이프라인 (업데이트 버전: 도메인 무관, 항상 모델/코드 생성):
# 1) embedder        : 문서 파싱/임베딩/벡터스토어 생성 및 retriever 세팅
# 2) summary_node    : 업로드 즉시 논문 요약 (리팩토링된 summarizer_agent 사용)
# 3) classify_node   : 기술 도메인 분류 (참고용. 분기에는 사용하지 않음 / 유지 선택사항)
# 4) model_extractor : 원문에서 모델 설명 섹션을 추출/분석하여 모델 정보 JSON 생성
# 5) base_code       : 모델 정보 기반으로 TensorFlow(Keras) base code 생성 (LLM-fallback)
# 6) qa_node         : 사용자 질문이 들어온 경우에만 실행 (리팩토링된 qa_agent 사용)
# ------------------------------------------------------------

from __future__ import annotations
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Any, Optional

# 기존 서비스
from services.summarizer import summarizer_agent, qa_agent
from services.classifier import classifier_agent
from services.embedder import embedder

# ✅ 새로 추가된 에이전트
# - run_model_extractor(raw_text) -> {"model_name":..., "components":[...], "description":...}
# - generate_base_code(model_name, components, description) -> str (Keras 단일 파일)
from services.model_extractor_agent import run_model_extractor
from services.base_code_generator_agent import generate_base_code

import logging
logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    user_input: str
    raw_text: str
    raw_texts: List[str]  # 여러 청크 텍스트 (요약 품질↑)
    chunks: list
    vectorstore: any
    retriever: any 
    meta: Dict[str, Any]  # 문서 메타: {"title": "...", "source": "..."} 등

    chat_history: Annotated[list, "Chat History"]
    summary: str
    domain: str
    answer: str
    top_k: int

    # ===== 신규: 모델/코드 결과 =====
    used_model: Optional[str]            # 예: "LSTM Autoencoder", "Vision Transformer"
    base_code: Optional[str]             # Keras base code (텍스트)

    # ===== 내부 전달/디버깅용 (선택) =====
    _model_components: List[str]         # 예: ["CNN backbone","Transformer encoder","MLP head"]
    _model_description: str              # 모델 요약 설명 문자열



# ------------------------------------------------------------
# 노드 함수 정의
# ------------------------------------------------------------
def model_extractor_node(state: AgentState) -> AgentState:
    """
    (도메인 무관) raw_text에서 모델 설명 섹션을 추출 → LLM으로 모델 정보를 구조화.
    결과:
      - state.used_model
      - state._model_components
      - state._model_description
    """
    info = run_model_extractor(state.get("raw_text", "") or "")
    state["used_model"] = info.get("model_name")
    state["_model_components"] = info.get("components", []) or []
    state["_model_description"] = info.get("description", "") or ""
    logger.info(f"[model_extractor] model={state['used_model']} comps={len(state['_model_components'])}")

    return state


def base_code_node(state: AgentState) -> AgentState:
    """
    모델명/구성/설명 기반으로 TensorFlow(Keras) base code를 LLM으로 생성.
    결과:
      - state.base_code
    """
    code = generate_base_code(
        model_name=state.get("used_model"),
        components=state.get("_model_components", []),
        description=state.get("_model_description", "")
    )
    state["base_code"] = code
    logger.info(f"[base_code] length={len(code) if code else 0}")

    return state



def build_graph():
    graph = StateGraph(AgentState)


    # 1) 노드 등록
    # embedder: 업로드된 문서에서 텍스트/청크/임베딩/벡터스토어/리트리버 생성
    graph.add_node("embedder", embedder)

    # summary_node: 리팩토링된 summarizer_agent
    # - 기대 입력: raw_texts 또는 raw_text (둘 중 하나), meta(선택)
    # - 출력: {"summary": "..."}
    graph.add_node("summary_node", summarizer_agent)

    # classify_node: 도메인 분류 에이전트
    # - 기대 입력: summary 또는 raw_texts/raw_text
    # - 출력: {"domain": "..."} (구현에 따라 다를 수 있음)
    graph.add_node("classify_node", classifier_agent)

    graph.add_node("model_extractor", model_extractor_node)  # ✅ 신규
    graph.add_node("base_code_gen", base_code_node)              # ✅ 신규


    # qa_node: 리팩토링된 qa_agent
    # - 기대 입력: user_input, retriever, (선택)top_k
    # - 출력: {"answer": "..."}
    graph.add_node("qa_node", qa_agent)


    # 2) 진입점
    graph.set_entry_point("embedder")

    # 3) 엣지 (직렬 흐름)
    # 업로드 즉시: 임베딩 -> 요약 -> 분류
    graph.add_edge("embedder", "summary_node")
    graph.add_edge("summary_node", "classify_node")
    graph.add_edge("classify_node", "model_extractor")   # 항상 모델 추출 실행
    graph.add_edge("model_extractor", "base_code_gen")       # 이어서 base code 생성

    # QA 노드는 조건부로 실행
    def should_run_qa(state: AgentState) -> bool:
        return "user_input" in state and state["user_input"] is not None

    graph.add_conditional_edges("base_code_gen", should_run_qa, {
        True: "qa_node",
        False: END,
    })

    graph.add_edge("qa_node", END)
    return graph.compile()