from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException  # ☆ HTTPException 추가
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from backend.database import get_db
from backend import models

from services.file_reader import file_reader
from services.graph_builder import build_graph
from services.summarizer import qa_agent  # ☆ 업로드+질문 엔드포인트에서 사용
from services.retriever_cache import set_retriever  # ☆ retriever 캐시 등록

import os
import shutil
from datetime import datetime

router = APIRouter()
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ☆ 그래프는 서버 구동 시 1회 컴파일 → 매 요청마다 재컴파일 비용 절약
_GRAPH = build_graph()


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    question: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    업로드 + 즉시 질문까지 한번에 처리.
    1) 파일 저장
    2) file_reader로 raw_text/meta 준비
    3) 그래프 실행(임베딩→요약→분류)
    4) Document 저장
    5) retriever 캐시에 보관
    6) qa_agent로 질문 답변 생성
    7) QA 히스토리 저장
    """
    # 사용자 하드코딩 (id=1) — 운영에서는 인증 연동
    user = db.query(models.User).filter_by(id=1).first()
    if not user:
        # 존재하지 않으면 생성 (초기 세팅 편의)
        user = models.User(id=1, email="test@example.com")
        db.add(user)
        db.commit()
        db.refresh(user)

    # 💡 중복 문서 체크(같은 사용자, 같은 파일명)
    existing_doc = (
        db.query(models.Document)
        .filter(models.Document.user_id == user.id)
        .filter(models.Document.filename == file.filename)
        .first()
    )
    if existing_doc:
        # 이미 분석된 문서라면, retriever 캐시가 비어있을 수 있으므로
        # 여기서 캐시 복구는 하지 않고 문서 정보만 반환 (질문은 /qa/ask_existing에서 복구 로직 처리 권장)
        return {
            "message": "File already uploaded.",
            "document_id": existing_doc.id,
            "summary": existing_doc.summary,
            "domain": existing_doc.domain,
        }

    # 1) 파일 저장
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2) 파일 읽기 (raw_text/meta/documents 생성)
    file_state = file_reader({"file": file_path})
    raw_text = file_state.get("raw_text", "")
    if not raw_text:
        raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출하지 못했습니다.")

    # 3) 그래프 실행 (임베딩 → 요약 → 분류)
    # ☆ 핵심: raw_text 뿐 아니라 meta도 포함된 state를 그대로 전달
    result = _GRAPH.invoke(file_state)

    summary = result.get("summary", "") or ""
    domain = result.get("domain", "") or ""
    retriever = result.get("retriever")
    vectorstore = result.get("vectorstore")

    # 4) Document 저장
    document = models.Document(
        user_id=user.id,
        filename=file.filename,
        file_path=file_path,
        summary=summary,
        domain=domain,
        uploaded_at=datetime.utcnow(),
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # 5) retriever 캐시에 보관 → /qa/ask_existing에서 재사용
    if retriever and vectorstore:
        set_retriever(document.id, retriever, vectorstore)

    # 6) 업로드와 동시에 받은 질문에 답변 생성 (빠르게: qa_agent만 호출)
    qa_out = qa_agent.invoke({
        "user_input": question,
        "retriever": retriever,
        "top_k": 5,
    }) if retriever else {"answer": "retriever가 없어 즉시 QA를 수행할 수 없습니다."}
    answer = qa_out.get("answer", "")

    # 7) QA 히스토리 저장
    qa_entry = models.QAHistory(
        document_id=document.id,
        question=question,
        answer=answer,
        created_at=datetime.utcnow(),
    )
    db.add(qa_entry)
    db.commit()

    # 8) 최종 응답
    return JSONResponse(
        content={
            "filename": file.filename,
            "summary": summary,
            "domain": domain,
            "answer": answer,
            "document_id": document.id,
        }
    )


@router.get("/documents")
def get_documents(db: Session = Depends(get_db)):
    """
    app.py가 기대하는 필드(id, filename, domain, summary, uploaded_at)를 반환.
    """
    documents = db.query(models.Document).all()
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "domain": doc.domain,
            "summary": doc.summary,
            "uploaded_at": doc.uploaded_at,
        }
        for doc in documents
    ]


@router.delete("/documents/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """
    문서 삭제 시 관련 QA 히스토리도 함께 삭제.
    """
    document = (
        db.query(models.Document).filter(models.Document.id == document_id).first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # QA 레코드 삭제
    db.query(models.QAHistory).filter(
        models.QAHistory.document_id == document_id
    ).delete()

    # 문서 삭제
    db.delete(document)
    db.commit()

    return {"message": "Document deleted successfully"}


@router.post("/documents/analyze_only")
async def analyze_document_only(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    업로드 → 분석만 수행 (질문 없음)
    1) 파일 저장
    2) file_reader로 state 생성
    3) 그래프 실행(임베딩→요약→분류)
    4) Document 저장
    5) retriever 캐시에 등록
    """
    # 사용자 하드코딩 (id=1)
    user = db.query(models.User).filter_by(id=1).first()
    if not user:
        user = models.User(id=1, email="test@example.com")
        db.add(user)
        db.commit()
        db.refresh(user)

    # 중복 문서 검사
    existing_doc = (
        db.query(models.Document)
        .filter(models.Document.user_id == user.id)
        .filter(models.Document.filename == file.filename)
        .first()
    )
    if existing_doc:
        # 이미 분석된 문서. retriever 캐시는 없을 수 있지만 여기서는 문서 정보만 반환.
        return {
            "message": "File already uploaded.",
            "document_id": existing_doc.id,
            "summary": existing_doc.summary,
            "domain": existing_doc.domain,
        }

    # 1) 파일 저장
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2) 파일 읽기
    file_state = file_reader({"file": file_path})
    raw_text = file_state.get("raw_text", "")
    if not raw_text:
        raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출하지 못했습니다.")

    # 3) 그래프 실행 (meta 포함 state 전체 전달)
    result = _GRAPH.invoke(file_state)

    summary = result.get("summary", "") or ""
    domain = result.get("domain", "") or ""
    retriever = result.get("retriever")
    vectorstore = result.get("vectorstore")

    # 4) Document 저장
    document = models.Document(
        user_id=user.id,
        filename=file.filename,
        file_path=file_path,
        summary=summary,
        domain=domain,
        uploaded_at=datetime.utcnow()
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # 5) retriever 캐시에 등록
    if retriever and vectorstore:
        set_retriever(document.id, retriever, vectorstore)

    return {
        "message": "Document analyzed.",
        "document_id": document.id,
        "summary": summary,
        "domain": domain,
    }
