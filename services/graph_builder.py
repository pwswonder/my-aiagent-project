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


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("embedder", embedder)
    graph.add_node("summary_node", summarizer_agent)
    graph.add_node("classify_node", classifier_agent)
    graph.add_node("qa_node", qa_agent)

    graph.set_entry_point("embedder")
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