from fastapi import APIRouter, UploadFile, File, Form, Depends, Query, HTTPException
from fastapi.responses import JSONResponse
import os, shutil
from backend.database import SessionLocal, get_db
from sqlalchemy.orm import Session
from backend import models, schemas, crud
from typing import List
from services.file_reader import file_reader
from services.graph_builder import build_graph
from langgraph.graph import StateGraph


router = APIRouter()
UPLOAD_FOLDER = "uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.get("/qa/history/docs", response_model=List[schemas.DocumentOut])
def get_docs_by_user(
    user_id: int = Query(..., description="사용자 ID"), db: Session = Depends(get_db)
):
    return db.query(models.Document).filter(models.Document.user_id == user_id).all()


@router.post("/qa/submit")
async def handle_question(file: UploadFile = File(...), question: str = Form(...)):
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ai_response = f"📘 질문 '{question}'에 대한 응답입니다 (파일: {file.filename})"

    db = SessionLocal()
    document = models.Document(user_id=1, title=file.filename, file_path=file_location)
    db.add(document)
    db.commit()
    db.refresh(document)

    qa = models.QAHistories(
        document_id=document.id, user_input=question, ai_answer=ai_response
    )
    db.add(qa)
    db.commit()
    return JSONResponse(content={"answer": ai_response})


@router.get("/qa/history/{document_id}", response_model=List[schemas.QAHistoryOut])
def get_qa_history(document_id: int, db: Session = Depends(get_db)):
    return (
        db.query(models.QAHistories)
        .filter(models.QAHistories.document_id == document_id)
        .all()
    )

@router.post("/qa/ask_existing")
def ask_existing_document_question(
    payload: schemas.ExistingDocQARequest,  # document_id + question
    db: Session = Depends(get_db)
):
    # 1. 문서 존재 여부 확인
    document = crud.get_document_by_id(db, payload.document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    # 2. 문서 내용 읽기
    try:
        content = file_reader(document.path)  # PDF에서 텍스트 추출
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 읽기 실패: {e}")

    # 3. LangGraph 에이전트 실행
    try:
        graph = build_graph()
        result = graph.invoke({"document": content, "question": payload.question})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"에이전트 실행 오류: {e}")

    if "answer" not in result:
        raise HTTPException(status_code=500, detail="답변 생성 실패")

    # 4. QA 히스토리 DB 저장
    qa_entry = schemas.QACreate(
        document_id=payload.document_id,
        question=payload.question,
        answer=result["answer"]
    )
    crud.save_qa_history(db, qa=qa_entry)

    # 5. 응답 반환
    return {"answer": result["answer"]}
