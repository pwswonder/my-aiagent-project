from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

from services.summarizer import summarizer_agent, qa_agent
from services.classifier import classifier_agent
from services.embedder import embedder

class AgentState(TypedDict, total=False):
    user_input: str
    raw_text: str
    chunks: list
    vectorstore: any
    retriever: any 
    chat_history: Annotated[list, "Chat History"]
    summary: str
    domain: str
    answer: str

# def route(state: AgentState) -> str:
#     user_input = state["user_input"].lower()
#     if "요약" in user_input:
#         return "summary_node"
#     elif "분류" in user_input or "도메인" in user_input:
#         return "classify_node"
#     elif "질문" in user_input or "무엇" in user_input:
#         return "qa_node"
#     else:
#         return "summary_node"


def build_graph():
    graph = StateGraph(AgentState)

    # 각 노드 추가
    graph.add_node("embedder", embedder)
    graph.add_node("summary_node", summarizer_agent)
    graph.add_node("classify_node", classifier_agent)
    graph.add_node("qa_node", qa_agent)

    # 실행 흐름 구성 (순차 실행)
    graph.set_entry_point("embedder")
    graph.add_edge("embedder", "summary_node")
    graph.add_edge("summary_node", "classify_node")
    graph.add_edge("classify_node", "qa_node")
    graph.add_edge("qa_node", END)

    return graph.compile()