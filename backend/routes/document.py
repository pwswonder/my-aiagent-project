from fastapi import APIRouter, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from backend.database import get_db
from backend import models

from services.file_reader import file_reader
from services.graph_builder import build_graph

import os
import shutil
from datetime import datetime

router = APIRouter()
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    question: str = Form(...),
    db: Session = Depends(get_db),
):
    # 사용자 하드코딩 (1번 사용자)
    user = db.query(models.User).filter_by(id=1).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 💡 중복 문서 체크
    existing_doc = (
        db.query(models.Document)
        .filter(models.Document.user_id == user.id)
        .filter(models.Document.filename == file.filename)
        .first()
    )
    if existing_doc:
        print("✅ 중복 문서 발견: 기존 문서 ID =", existing_doc.id)
        return {
            "message": "File already uploaded.",
            "document_id": existing_doc.id,
            "summary": existing_doc.summary,
            "domain": existing_doc.domain,
        }

    # 1. 파일 저장
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2. 사용자 정보 (id=1 고정, 없으면 생성)
    user = db.query(models.User).filter_by(id=1).first()
    if not user:
        user = models.User(id=1, email="test@example.com")
        db.add(user)
        db.commit()
        db.refresh(user)

    # 3. 텍스트 추출
    file_state = file_reader({"file": file_path})
    raw_text = file_state["raw_text"]

    # 4. LangGraph 실행 (StateGraph 기반)
    graph = build_graph()
    result = graph.invoke({"raw_text": raw_text, "user_input": question})

    # 5. 결과 추출
    summary = result.get("summary", "")
    domain = result.get("domain", "")
    answer = result.get("answer", "")
    print("[DEBUG] LangGraph result:", result)  # 디버깅용

    # 6. Document 저장
    document = models.Document(
        user_id=user.id,
        filename=file.filename,
        summary=summary,
        domain=domain,
        uploaded_at=datetime.utcnow(),
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # 7. QA 히스토리 저장
    qa_entry = models.QAHistory(
        document_id=document.id,
        question=question,
        answer=answer,
        created_at=datetime.utcnow(),
    )
    db.add(qa_entry)
    db.commit()

    # 8. 최종 응답
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
    document = (
        db.query(models.Document).filter(models.Document.id == document_id).first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # 수동으로 QA 레코드 삭제
    db.query(models.QAHistory).filter(
        models.QAHistory.document_id == document_id
    ).delete()

    # 문서 삭제
    db.delete(document)
    db.commit()

    return {"message": "Document deleted successfully"}
