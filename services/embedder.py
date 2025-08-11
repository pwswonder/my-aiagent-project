# services/embedder.py
from __future__ import annotations

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from typing import TypedDict, List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

from langsmith import traceable



class EmbedState(TypedDict, total=False):
    """
    그래프의 embedder 노드 입출력 상태 정의.
    total=False: 모든 키는 optional (동적 파이프라인 호환)
    """
    # 입력
    raw_text: str                      # 업로드/파싱된 전체 텍스트
    meta: Dict[str, Any]               # 문서 메타 (예: {"title": "...", "source": "filename.pdf"})

    # 출력
    raw_texts: List[str]               # 요약 모델이 사용할 청크 텍스트 목록
    chunks: List[str]                  # (과거 호환) 단순 문자열 청크
    vectorstore: FAISS                 # 임베딩된 벡터스토어
    retriever: Any                     # 검색기 (as_retriever)
    # 옵션
    top_k: int                         # QA 검색 문서 수 (기본값 내부에서 설정)

@traceable  # ★ 이 1줄만 추가

def _build_chunks(raw_text: str, meta: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """
    텍스트를 섹션-친화적으로 청크화하여,
    [ {"text": str, "metadata": {...}}, ... ] 형태로 반환.
    - 1차: MarkdownHeaderTextSplitter로 섹션 경계 보존
    - 2차: RecursiveCharacterTextSplitter로 길이 맞춤
    """
    meta = meta or {}
    # 1) 섹션 인식: 논문이 꼭 마크다운은 아니어도, 헤더 패턴(#, ##) 혹은 우리가 나중에 전처리해줄 수도 있음
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "section"), ("##", "subsection")],
        strip_headers=False,
    )
    try:
        header_docs = header_splitter.split_text(raw_text)
        base_sections = [{"text": d.page_content, "metadata": {"section": d.metadata.get("header", "")}} for d in header_docs]
    except Exception:
        # 헤더가 없거나 실패하면 전체를 하나의 섹션으로 취급
        base_sections = [{"text": raw_text, "metadata": {"section": "whole_document"}}]

    # 2) 길이 맞춤: 한글 기준 900~1200자 권장(= 대략 350~500 토큰)
    body_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # ← 기존 500보다 늘려 맥락 보존
        chunk_overlap=120,    # ← 문장 경계 부드럽게
        separators=["\n\n", "\n", " ", ""],  # 문단→문장→공백→문자 단위
    )

    results: List[Dict[str, Any]] = []
    for sec_id, sec in enumerate(base_sections):
        pieces = body_splitter.split_text(sec["text"])
        for i, piece in enumerate(pieces):
            md = {
                "source": meta.get("source") or meta.get("title") or "N/A",
                "title": meta.get("title", "N/A"),
                "section": sec.get("metadata", {}).get("section") or sec.get("section") or "N/A",
                "chunk_id": f"{sec_id}-{i}",
            }
            results.append({"text": piece, "metadata": md})
    return results

@traceable  # ★ 이 1줄만 추가
def embedder(state: EmbedState) -> EmbedState:
    """
    1) 텍스트를 섹션-보존 방식으로 청크화
    2) Azure OpenAI 임베딩으로 FAISS 벡터스토어 구성
    3) retriever 생성 (MMR/TopK 설정)
    4) 요약용 raw_texts도 함께 반환
    """
    # ---- 입력 파싱 ----
    raw_text = (state.get("raw_text") or "").strip()
    if not raw_text:
        return {**state, "retriever": None, "vectorstore": None, "chunks": [], "raw_texts": []}

    meta: Dict[str, Any] = state.get("meta", {}) or {}

    # ---- 1) 청크 생성 (섹션 보존 + 길이 맞춤) ----
    chunk_dicts = _build_chunks(raw_text, meta=meta)  # [{"text":..., "metadata":...}, ...]
    # 요약 모델이 바로 쓸 수 있도록 문자열 리스트도 준비
    raw_texts: List[str] = [c["text"] for c in chunk_dicts]

    # ---- 2) 임베딩 모델 준비 ----
    # ☆ 중요: Azure에선 azure_deployment 파라미터 사용
    embedding_model = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AOAI_DEPLOY_EMBED_3_LARGE"),
        openai_api_version="2024-02-01",
        api_key=os.getenv("AOAI_API_KEY"),
        azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    )

    # ---- 3) 벡터스토어 구축 (메타데이터 포함) ----
    texts = [c["text"] for c in chunk_dicts]
    metadatas = [c["metadata"] for c in chunk_dicts]

    # faiss-cpu가 없을 때를 대비한 안내를 두고 싶다면 try/except로 감싸도 좋음
    vectorstore = FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)

    # ---- 4) 리트리버 구성 ----
    # 기본값: MMR(다양성) + 상위 5개
    # fetch_k는 후보군, k는 최종 반환 개수
    top_k = state.get("top_k", 5)
    retriever = vectorstore.as_retriever(
        search_type="mmr",          # "similarity" 보다 논문 QA에 안정적
        search_kwargs={
            "k": top_k,
            "fetch_k": max(20, top_k * 6),   # 후보군은 넉넉히
            "lambda_mult": 0.5,              # 다양성(0~1), 0.5 정도 중립
        },
    )

    # ---- 5) 반환 ----
    # - 기존 호환을 위해 "chunks"는 문자열 리스트로 유지
    # - 새 summarizer가 이걸 바로 활용할 수 있도록 "raw_texts" 제공
    return {
        **state,
        "raw_texts": raw_texts,     # 새 summarizer는 이걸 우선 사용
        "chunks": raw_texts,        # 기존 호환(동일 내용)
        "vectorstore": vectorstore,
        "retriever": retriever,
        "top_k": top_k,             # 상태에 보존(qa 노드에서 재사용)
    }
