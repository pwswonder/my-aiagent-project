# backend/services/retriever_cache.py
# 문서별 retriever/vectorstore를 메모리에 보관 (개발/PoC 용)
# 운영에서는 디스크에 FAISS 저장 또는 외부 벡터DB 권장.

from typing import Dict, Any, Optional

_RETRIEVER_CACHE: Dict[int, Dict[str, Any]] = {}


def set_retriever(doc_id: int, retriever: Any, vectorstore: Any) -> None:
    _RETRIEVER_CACHE[doc_id] = {"retriever": retriever, "vectorstore": vectorstore}


def get_retriever(doc_id: int) -> Optional[Any]:
    item = _RETRIEVER_CACHE.get(doc_id)
    return item.get("retriever") if item else None


def has_retriever(doc_id: int) -> bool:
    return doc_id in _RETRIEVER_CACHE


def clear_retriever(doc_id: int) -> None:
    _RETRIEVER_CACHE.pop(doc_id, None)
