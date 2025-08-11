# services/file_reader.py
from langchain_community.document_loaders import PyMuPDFLoader
from typing import TypedDict, List, Dict, Any
import os


class DocState(TypedDict, total=False):
    file: str               # 입력: 파일 경로 (절대경로 or 상대경로)
    documents: List[Any]    # 출력: page 단위 문서 객체 리스트 (LangChain Document)
    raw_text: str           # 출력: 전체 텍스트 결합 버전
    meta: Dict[str, Any]    # 출력: 문서 메타데이터 (title, source 등)


def file_reader(state: DocState) -> DocState:
    """
    PDF 파일 경로를 받아서,
    - page 단위 문서를 추출하고 (LangChain 문서 객체 리스트)
    - 각 페이지의 텍스트를 결합하여 raw_text 생성
    - embedder/summarizer/QA 에이전트들이 사용할 수 있도록 준비된 상태 반환
    """
    # ---- 입력 확인 ----
    file_path = state.get("file")
    if not file_path or not os.path.exists(file_path):
        return {**state, "raw_text": "", "documents": [], "meta": {}}

    # ---- PyMuPDF로 로드 ----
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()  # List[Document] ← 각 문서에 .page_content, .metadata 있음

    # ---- 메타데이터 생성 ----
    file_name = os.path.basename(file_path)
    title = os.path.splitext(file_name)[0]

    # ---- 전체 텍스트 합치기 ----
    # 각 페이지 구분용 "\n\n--- Page N ---\n\n" 같은 포맷을 넣을 수도 있음
    raw_text = "\n".join([doc.page_content for doc in documents])

    # ---- 메타 정보 주입 ----
    for idx, doc in enumerate(documents):
        doc.metadata["source"] = file_name      # 원본 파일명
        doc.metadata["title"] = title           # 파일명에서 확장자 제거
        doc.metadata["page"] = idx + 1          # 1-based 페이지 번호

    return {
        **state,
        "documents": documents,
        "raw_text": raw_text,
        "meta": {
            "title": title,
            "source": file_name,
        }
    }
