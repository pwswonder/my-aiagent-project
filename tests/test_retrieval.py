from paper_agent_v2.parser import PaperChunk
from paper_agent_v2.retrieval import HybridRetriever


def test_rrf_prefers_matching_architecture_evidence() -> None:
    chunks = [
        PaperChunk("a", 1, "intro", "paragraph", "background and motivation"),
        PaperChunk("b", 4, "method", "paragraph", "multi head attention architecture with eight heads"),
    ]
    hits = HybridRetriever(chunks).search("attention eight heads", limit=1)
    assert hits[0].chunk.id == "b"
    assert hits[0].chunk.page == 4
