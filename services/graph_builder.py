# backend/services/graph_builder.py
# ------------------------------------------------------------
# LangGraph 파이프라인:
# 1) embedder  : 문서 파싱/임베딩/벡터스토어 생성 및 retriever 세팅
# 2) summary   : 업로드 즉시 논문 요약 (리팩토링된 summarizer_agent 사용)
# 3) classify  : 기술 도메인 분류
# 4) qa        : 사용자 질문이 들어온 경우에만 실행 (리팩토링된 qa_agent 사용)
# ------------------------------------------------------------

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Any

from services.summarizer import summarizer_agent, qa_agent
from services.classifier import classifier_agent
from services.embedder import embedder

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

    # QA 노드는 조건부로 실행
    def should_run_qa(state: AgentState) -> bool:
        return "user_input" in state and state["user_input"] is not None

    graph.add_conditional_edges("classify_node", should_run_qa, {
        True: "qa_node",
        False: END,
    })

    graph.add_edge("qa_node", END)
    return graph.compile()